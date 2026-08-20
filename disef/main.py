import os
import sys
import gc

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from peft import LoraConfig, get_peft_model

from data import get_data_loader, get_synth_train_data_loader
from utils import fix_random_seeds
from models.qwen.src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from qwen_utils import get_image_embedding, get_acc
from util_data import SUBSET_NAMES, TEMPLATES_SMALL
from utils import get_dataset_name_for_template, cosine_scheduler

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
    lamda,
    lr_schedule,
    writer,
    device,
    dataset="dtd",
    logit_scale=15,
):
    model.model.train()

    for real_images, real_labels in fewshot_train_loader:

        if step < len(lr_schedule):
            for param_group in opt_h.param_groups:
                param_group["lr"] = lr_schedule[step]

        step += 1

        synth_batch = next(synth_iter)
        synth_images, synth_labels = synth_batch[0], synth_batch[1]

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

            loss = (lamda * loss_real) + ((1 - lamda) * loss_synth)
            
        loss_value = loss.item()

        scaler.scale(loss).backward()
        scaler.step(opt_h)
        scaler.update()

        writer.add_scalar("Loss/Total", loss_value, step)
        writer.add_scalar("Loss/Real", loss_real.item(), step)
        writer.add_scalar("Loss/Synth", loss_synth.item(), step)

        del real_images, real_labels, real_imgs_embedding, logits_real_all, loss_real
        del synth_images, synth_labels, synth_imgs_embedding, logits_synth_all, loss_synth
        del loss, mu

    return step

def main():
    fix_random_seeds(22)

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
    lamda = 0.8

    fewshot_train_loader, test_loader = get_data_loader(
        real_train_data_dir="",
        real_test_data_dir="",
        dataset=dataset,
        bs=batch_size,
        eval_bs=eval_batch_size,
        n_img_per_cls=n_samples_per_class,
        model_type=model_type
    )

    synth_train_loader = get_synth_train_data_loader(
        synth_train_data_dir=synth_train_data_dir,
        bs=batch_size,
        n_img_per_cls=n_synth_per_class,
        dataset=dataset,
        model_type=model_type
    )

    exp_name = f"{dataset}_{n_samples_per_class}shot_{n_synth_per_class}synth_lamda{lamda}_datadream"

    print(f"Number of few-shot training samples: {len(fewshot_train_loader.dataset)}")
    print(f"Number of synthetic training samples: {len(synth_train_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")
    print(exp_name)

    log_dir_path = f"runs/{exp_name}"
    ckpt_path = f"checkpoints/{exp_name}"

    os.makedirs(log_dir_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)

    model = Qwen3VLEmbedder(model_name_or_path="./models/Qwen3-VL-Embedding-2B")

    lora_config = LoraConfig(
        r=16,             
        lora_alpha=32,      
        target_modules=["q_proj", "v_proj", "qkv", "proj"],
        lora_dropout=0.1,
        task_type="FEATURE_EXTRACTION"
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
        start_warmup_value=1e-6)

    step = 0
    best_test_acc = 0.0
    synth_iter = get_infinite_iter(synth_train_loader)


    for epoch in range(n_epochs):
        step = train_one_epoch(
            model=model, 
            opt_h=opt_h, 
            scaler=scaler, 
            step=step, 
            fewshot_train_loader=fewshot_train_loader,
            synth_iter=synth_iter,
            lamda=lamda,
            lr_schedule=lr_schedule,
            writer=writer, 
            device=device, 
            dataset=dataset,
            logit_scale=logit_scale
        )

        model.model.eval()
        with torch.no_grad():
            mu_eval = get_prototype(model, dataset)
            mu_eval = mu_eval.to(device)
            train_acc = get_acc(model, fewshot_train_loader, mu_eval, device)
            test_acc = get_acc(model, test_loader, mu_eval, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_path = os.path.join(ckpt_path, "best_disef_model")
            model.model.save_pretrained(best_model_path)

        writer.add_scalar("Epoch/Train_Accuracy", train_acc, global_step=epoch)
        writer.add_scalar("Epoch/Eval_Accuracy", test_acc, global_step=epoch)
        
        print(f"Epoch {epoch} | Train ACC: {train_acc*100:.2f}% | Test ACC: {test_acc*100:.2f}%")

        gc.collect()
    writer.close()
    
if __name__ == "__main__":
    main()