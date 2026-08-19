import os
import sys
import gc
import faiss
import numpy as np
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from peft import LoraConfig, get_peft_model

from data import get_data_loader, get_synth_train_data_loader
from utils import fix_random_seeds
from models.qwen.src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from qwen_utils import get_image_embedding, get_acc
from util_data import SUBSET_NAMES, TEMPLATES_SMALL
from utils import get_dataset_name_for_template, cosine_scheduler

class IndexedDataset(Dataset):
    def __init__(self, dataset, is_real):
        self.dataset = dataset
        self.is_real = is_real

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if isinstance(data, tuple):
            return (*data[:2], self.is_real, idx)
        return data, self.is_real, idx

def indexed_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    is_reals = [item[2] for item in batch]
    idxs = [item[3] for item in batch]
    return images, labels, is_reals, idxs

def extract_flattened_images(dataset):
    features = []
    print("Extracting flattened images for FAISS clustering...")
    for i in range(len(dataset)):
        img = dataset[i][0] 
        if isinstance(img, Image.Image):
            arr = np.array(img.resize((256, 256), Image.BICUBIC), dtype=np.float32) / 255.0
        elif isinstance(img, torch.Tensor):
            arr = F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bicubic').squeeze(0)
            arr = arr.permute(1, 2, 0).numpy()
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
        features.append(arr.flatten())
    return np.vstack(features)

def cluster_training_datasets_with_faiss(real_dataset, synth_dataset, num_centroids=10, niter=20):
    print("Clustering training datasets with FAISS...")
    
    real_data = extract_flattened_images(real_dataset)
    synth_data = extract_flattened_images(synth_dataset)

    all_data = np.vstack([synth_data, real_data])
    D = all_data.shape[1]

    kmeans = faiss.Kmeans(d=D, k=num_centroids, niter=niter, verbose=True, gpu=True)
    kmeans.train(all_data)
    
    _, assignments = kmeans.index.search(all_data, 1)
    assignments = assignments.reshape(-1)
    
    num_synth = len(synth_dataset)
    
    idx_to_region = {}

    for i, cluster_idx in enumerate(assignments):
        if i < num_synth:
            idx_to_region[(False, i)] = cluster_idx
        else:
            real_idx = i - num_synth
            idx_to_region[(True, real_idx)] = cluster_idx

    print(f"[INFO] Clustering completed. Number of clusters: {num_centroids}.")
    return idx_to_region

def get_infinite_iter(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def get_prototype(model, dataset):
    class_names = SUBSET_NAMES[dataset]
    dataset_name = get_dataset_name_for_template(dataset)

    all_mus = []

    for class_name in class_names:
        class_texts = [
            {"text": template.format(dataset_name, class_name) + '.'}
            for template in TEMPLATES_SMALL
        ]

        class_embs = model.process(class_texts)
        class_embs = F.normalize(class_embs, dim=-1)

        mu = class_embs.mean(dim=0)
        mu = F.normalize(mu, dim=-1)

        all_mus.append(mu)

    return torch.stack(all_mus)

def train_one_epoch(
    model,
    opt_h,
    scaler,
    step,
    fewshot_train_loader,
    synth_iter,
    lr_schedule,
    writer,
    device,
    dataset,
    logit_scale,
    idx_to_region,
    num_centroids,
    lam_real=0.8,
    lam_synth=0.2,
    lam_dis=0.1,
    lam_rob=0.1,
    g_factor=1.0,
):
    model.model.train()

    for real_images, real_labels, _, real_idxs in fewshot_train_loader:

        if step < len(lr_schedule):
            for param_group in opt_h.param_groups:
                param_group["lr"] = lr_schedule[step]

        step += 1

        synth_batch = next(synth_iter)
        synth_images, synth_labels, _, synth_idxs = synth_batch

        real_labels = real_labels.to(device)
        synth_labels = synth_labels.to(device)

        opt_h.zero_grad(set_to_none=True)

        mu = get_prototype(model, dataset)

        with torch.amp.autocast("cuda"):
            real_imgs_embedding = get_image_embedding(model, real_images)
            synth_imgs_embedding = get_image_embedding(model, synth_images)

            logits_real_all = logit_scale * (real_imgs_embedding @ mu.t())
            logits_synth_all = logit_scale * (synth_imgs_embedding @ mu.t())
            loss_real = F.cross_entropy(logits_real_all, real_labels)
            loss_synth = F.cross_entropy(logits_synth_all, synth_labels)
            region_real_in_batch = [[] for _ in range(num_centroids)]
            region_syn_in_batch  = [[] for _ in range(num_centroids)]

            for b_idx, g_idx in enumerate(real_idxs):
                region_idx = idx_to_region[(True, g_idx)]
                region_real_in_batch[region_idx].append(b_idx)

            for b_idx, g_idx in enumerate(synth_idxs):
                region_idx = idx_to_region[(False, g_idx)]
                region_syn_in_batch[region_idx].append(b_idx)

            total_discrepancy_loss = 0.0
            total_robustness_loss = 0.0

            for region_idx in range(num_centroids):
                real_indices = region_real_in_batch[region_idx]
                syn_indices  = region_syn_in_batch[region_idx]
                num_real = len(real_indices)
                num_syn  = len(syn_indices)

                if num_real > 0 and num_syn > 0:
                    pred_real = real_imgs_embedding[real_indices]
                    pred_syn  = synth_imgs_embedding[syn_indices]

                    pairwise_mse = F.mse_loss(
                        pred_real.unsqueeze(1),
                        pred_syn.unsqueeze(0),
                        reduction='none'
                    ).mean(dim=-1)
                    total_discrepancy_loss += pairwise_mse.sum() / (g_factor * num_real)

                if num_syn > 1 and num_real > 0:
                    pred_syn_region = synth_imgs_embedding[syn_indices]
                    pairwise_mse_syn = F.mse_loss(
                        pred_syn_region.unsqueeze(1),
                        pred_syn_region.unsqueeze(0),
                        reduction='none'
                    ).mean(dim=-1)
                    
                    i_upper = torch.triu_indices(num_syn, num_syn, offset=1)
                    mse_upper = pairwise_mse_syn[i_upper[0], i_upper[1]]
                    total_robustness_loss += mse_upper.sum() / (g_factor * num_syn)

            loss = (lam_real * loss_real) + (lam_synth * loss_synth) \
                   + (lam_dis * total_discrepancy_loss) \
                   + (lam_rob * total_robustness_loss)
            
        loss_value = loss.item()

        scaler.scale(loss).backward()
        scaler.step(opt_h)
        scaler.update()

        writer.add_scalar("Loss/Total", loss_value, step)
        writer.add_scalar("Loss/Real", loss_real.item(), step)
        writer.add_scalar("Loss/Synth", loss_synth.item(), step)
        if isinstance(total_discrepancy_loss, torch.Tensor):
            writer.add_scalar("Loss/Discrepancy", total_discrepancy_loss.item(), step)
        if isinstance(total_robustness_loss, torch.Tensor):
            writer.add_scalar("Loss/Robustness", total_robustness_loss.item(), step)

        del real_images, real_labels, real_imgs_embedding, logits_real_all, loss_real
        del synth_images, synth_labels, synth_imgs_embedding, logits_synth_all, loss_synth
        del loss, mu

    return step

def main():
    fix_random_seeds(22)
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "dtd"
    model_type = "qwen"
    n_samples_per_class = 16
    n_epochs = 20
    batch_size = 64
    eval_batch_size = 64
    logit_scale = 15
    synth_train_data_dir = "synthetic_data"
    n_synth_per_class = 64
    lam_real = 0.8
    lam_synth = 0.2
    lam_dis = 0.1
    lam_rob = 0.1
    num_centroids = 47

    fewshot_train_loader_raw, test_loader = get_data_loader(
        real_train_data_dir="", real_test_data_dir="",
        dataset=dataset, bs=batch_size, eval_bs=eval_batch_size,
        n_img_per_cls=n_samples_per_class, model_type=model_type
    )

    synth_train_loader_raw = get_synth_train_data_loader(
        synth_train_data_dir=synth_train_data_dir, bs=batch_size,
        n_img_per_cls=n_synth_per_class, dataset=dataset, model_type=model_type
    )

    real_dataset_indexed = IndexedDataset(fewshot_train_loader_raw.dataset, is_real=True)
    synth_dataset_indexed = IndexedDataset(synth_train_loader_raw.dataset, is_real=False)

    idx_to_region = cluster_training_datasets_with_faiss(
        real_dataset=real_dataset_indexed,
        synth_dataset=synth_dataset_indexed,
        num_centroids=num_centroids
    )

    fewshot_train_loader = DataLoader(
        real_dataset_indexed, batch_size=batch_size, shuffle=True,
        num_workers=fewshot_train_loader_raw.num_workers,
        pin_memory=True, collate_fn=indexed_collate_fn
    )

    synth_train_loader = DataLoader(
        synth_dataset_indexed, batch_size=batch_size, shuffle=True,
        num_workers=synth_train_loader_raw.num_workers,
        pin_memory=True, collate_fn=indexed_collate_fn
    )

    exp_name = f"{dataset}_{n_samples_per_class}shot_{n_synth_per_class}synth_lam_real{lam_real}_lam_synth{lam_synth}_dis{lam_dis}_rob{lam_rob}_centroids{num_centroids}_protoaug"
    print(f"Number of few-shot training samples: {len(real_dataset_indexed)}")
    print(f"Number of synthetic training samples: {len(synth_dataset_indexed)}")
    print(exp_name)

    log_dir_path = f"runs/{exp_name}"
    ckpt_path = f"checkpoints/{exp_name}"
    os.makedirs(log_dir_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)

    model = Qwen3VLEmbedder(model_name_or_path="./models/Qwen3-VL-Embedding-2B")

    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "qkv", "proj"],
        lora_dropout=0.1, task_type="FEATURE_EXTRACTION"
    )

    model.model = get_peft_model(model.model, lora_config)
    model.model.gradient_checkpointing_enable() 
    model.model.print_trainable_parameters()

    writer = SummaryWriter(log_dir=log_dir_path)
    trainable_params =[p for p in model.model.parameters() if p.requires_grad]
    opt_h = torch.optim.AdamW(trainable_params, lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    niter_per_ep = len(fewshot_train_loader)
    lr_schedule = cosine_scheduler(
        base_value=1e-4,
        final_value=1e-6,
        epochs=n_epochs, 
        niter_per_ep=niter_per_ep, 
        warmup_epochs=2, 
        start_warmup_value=1e-6
    )

    step = 0
    best_test_acc = 0.0
    synth_iter = get_infinite_iter(synth_train_loader)

    g_factor = n_samples_per_class * len(SUBSET_NAMES[dataset])

    for epoch in range(n_epochs):
        step = train_one_epoch(
            model=model, opt_h=opt_h, scaler=scaler, step=step, 
            fewshot_train_loader=fewshot_train_loader,
            synth_iter=synth_iter, lr_schedule=lr_schedule,
            writer=writer, device=device, dataset=dataset,
            logit_scale=logit_scale, 
            idx_to_region=idx_to_region, 
            num_centroids=num_centroids,
            lam_real=lam_real, 
            lam_synth=lam_synth,
            lam_dis=lam_dis,
            lam_rob=lam_rob,
            g_factor=g_factor
        )

        model.model.eval()
        with torch.no_grad():
            mu_eval = get_prototype(model, dataset).to(device)
            train_acc = get_acc(model, fewshot_train_loader_raw, mu_eval, device)
            test_acc = get_acc(model, test_loader, mu_eval, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_path = os.path.join(ckpt_path, "best_real_finetuned_model")
            model.model.save_pretrained(best_model_path)

        writer.add_scalar("Epoch/Train_Accuracy", train_acc, global_step=epoch)
        writer.add_scalar("Epoch/Eval_Accuracy", test_acc, global_step=epoch)
        print(f"Epoch {epoch} | Train ACC: {train_acc*100:.2f}% | Test ACC: {test_acc*100:.2f}%")

        gc.collect()
    writer.close()
    
if __name__ == "__main__":
    main()