import os
import sys
import gc

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from flask import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from peft import LoraConfig, get_peft_model

from data import get_data_loader, get_synth_train_data_loader
from utils import fix_random_seeds
from models.qwen.src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from qwen_utils import get_image_embedding, get_acc
from utils import cosine_scheduler

def get_infinite_iter(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def get_prototype(model, prompt_path="data.json"):
    with open(prompt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_mus = []
    all_kappas = []

    for class_name in data.keys():
        class_texts = [{"text": prompts} for prompts in data[class_name]]

        class_embs = model.process(class_texts)
        class_embs = F.normalize(class_embs, dim=-1)

        mu = class_embs.mean(dim=0)
        D = mu.shape[0]
        R = mu.norm()

        mu = mu / R
        kappa = (R * (D - R**2)) / torch.clamp(1 - R**2, min=1e-6)

        all_mus.append(mu)
        all_kappas.append(kappa)

    all_mus = torch.stack(all_mus)
    all_kappas = torch.stack(all_kappas)

    return all_mus, all_kappas

@torch.no_grad()
def build_text_distribution_samples(mu, kappa, num_samples=30, kappa_scale=0.05, kappa_max=float('inf')):
    C, D = mu.shape
    device = mu.device
    dtype = mu.dtype
    kappa = (kappa * kappa_scale).view(C, 1, 1)
    
    eps = torch.randn((C, num_samples, D), device=device, dtype=dtype)
    dot_product = torch.bmm(eps, mu.unsqueeze(-1))
    mu_expanded = mu.unsqueeze(1)
    eps.sub_(dot_product * mu_expanded) 
    eps.div_(torch.sqrt(kappa + 1e-6))
    samples = eps.add_(mu_expanded)

    return F.normalize(samples, p=2, dim=-1)

# def logit_from_h_vectorized(logit_scale, image_feats, centroids, area_index, chosen_centroids):
#     logits_base = logit_scale * (image_feats @ centroids.t())
#     logits_samples = logit_scale * (image_feats @ chosen_centroids.t())
#     S = chosen_centroids.shape[0]
#     logits_all = logits_base.unsqueeze(0).repeat(S, 1, 1)
#     logits_all[:, :, area_index] = logits_samples.t()

#     return logits_all

# def compute_reg_vectorized(logit_scale, area_index, samples, feats_i, centroids):
#     if feats_i.shape[0] == 0:
#         return torch.tensor(0.0, device=feats_i.device)

#     sampled_centroids = samples[area_index]
#     logits_all = logit_from_h_vectorized(logit_scale, feats_i, centroids, area_index, sampled_centroids)
#     logits_flat = logits_all.reshape(-1, logits_all.size(-1))
#     labels_flat = torch.full((logits_flat.size(0),), area_index, device=feats_i.device, dtype=torch.long)

#     return F.cross_entropy(logits_flat, labels_flat)

def compute_reg_vectorized(logit_scale, area_index, samples, feats_i, centroids, logits_base=None):
    if feats_i.shape[0] == 0:
        return torch.tensor(0.0, device=feats_i.device)
    
    if logits_base is None:
        logits_base = logit_scale * (feats_i @ centroids.t())  # (N, C)

    sampled_centroids = samples[area_index]                     # (S, D)
    logits_samples = logit_scale * (feats_i @ sampled_centroids.t())  # (N, S)

    base_masked = logits_base.clone()
    base_masked[:, area_index] = float('-inf')
    base_max = base_masked.max(dim=-1, keepdim=True).values      # (N, 1)
    base_sumexp = torch.exp(base_masked - base_max).sum(dim=-1)  # (N,)
    term_fixed = (torch.log(base_sumexp + 1e-12) + base_max.squeeze(-1)).unsqueeze(1)  # (N,1)

    logsumexp_total = torch.logaddexp(term_fixed, logits_samples)  # (N, S)
    loss_per = logsumexp_total - logits_samples                    # (N, S)

    return loss_per.mean()

def compute_reg_batch(logit_scale, samples, feats, labels, centroids):
    # feats:    [N, D]
    # labels:   [N]
    # samples:  [C, S, D]
    # centroids:[C, D]
    N = feats.shape[0]
    if N == 0:
        return feats.sum() * 0.0

    base_logits = logit_scale * (feats @ centroids.t())          # [N, C]
    sampled = samples[labels]                                     # [N, S, D]
    sampled_logits = logit_scale * torch.einsum(                  # [N, S]
        "nd,nsd->ns", feats, sampled,
    )

    original_target = base_logits.gather(1, labels[:, None]).squeeze(1)  # [N]
    all_lse = torch.logsumexp(base_logits, dim=1)                        # [N]
    diff = original_target - all_lse
    other_lse = all_lse + torch.log1p(-torch.exp(diff))                  # [N]

    loss_per_sample = torch.logaddexp(other_lse[:, None], sampled_logits) - sampled_logits  # [N, S]
    loss_per_image = loss_per_sample.mean(dim=1)  # [N]

    unique_labels, counts = labels.unique(return_counts=True)
    count_per_image = counts[torch.searchsorted(unique_labels, labels)]  # [N]
    weight_per_image = 1.0 / count_per_image.float()
    num_classes = unique_labels.numel()

    loss = (loss_per_image * weight_per_image).sum() / num_classes
    return loss

def compute_consistency_reg(real_per_sample_loss, synth_per_sample_loss, real_labels, synth_labels, num_classes):
    if real_per_sample_loss.numel() == 0 or synth_per_sample_loss.numel() == 0:
        return real_per_sample_loss.sum() * 0.0

    diff = (real_per_sample_loss.unsqueeze(1) - synth_per_sample_loss.unsqueeze(0)).abs()

    same_class = real_labels.unsqueeze(1) == synth_labels.unsqueeze(0)  # [N_r, N_g] bool

    class_ids = real_labels.unsqueeze(1).expand(-1, synth_labels.shape[0])[same_class]  # [num_pairs]
    diff_masked = diff[same_class]  # [num_pairs]

    sum_per_class = torch.zeros(num_classes, device=diff.device, dtype=diff.dtype)
    count_per_class = torch.zeros(num_classes, device=diff.device, dtype=diff.dtype)
    sum_per_class.scatter_add_(0, class_ids, diff_masked)
    count_per_class.scatter_add_(0, class_ids, torch.ones_like(diff_masked))

    valid = count_per_class > 0
    eps_per_class = sum_per_class[valid] / count_per_class[valid]

    return eps_per_class.mean()

def train_one_epoch(
    model,
    prompt_path,
    opt_h,
    scaler,
    step,
    fewshot_train_loader,
    synth_iter,
    lamda1,
    lamda2,
    lamda3,
    lamda4,
    lr_schedule,
    writer,
    device,
    dataset="dtd",
    logit_scale=15,
):
    model.model.train()

    for real_images, real_labels in fewshot_train_loader:
        opt_h.zero_grad(set_to_none=True)

        if step < len(lr_schedule):
            for param_group in opt_h.param_groups:
                param_group["lr"] = lr_schedule[step]

        step += 1

        synth_batch = next(synth_iter)
        synth_images, synth_labels = synth_batch[0], synth_batch[1]

        real_labels = real_labels.to(device, non_blocking=True)
        synth_labels = synth_labels.to(device, non_blocking=True)

        mu, kappa = get_prototype(model, prompt_path)

        with torch.amp.autocast('cuda'):
            real_imgs_embedding = get_image_embedding(model, real_images)
            synth_imgs_embedding = get_image_embedding(model, synth_images)

            logits_real_all = logit_scale * (real_imgs_embedding @ mu.t())
            logits_synth_all = logit_scale * (synth_imgs_embedding @ mu.t())

            samples = build_text_distribution_samples(mu, kappa, num_samples=30)

            real_loss_per_sample = F.cross_entropy(logits_real_all, real_labels, reduction='none')   # [N_r]
            synth_loss_per_sample = F.cross_entropy(logits_synth_all, synth_labels, reduction='none') # [N_g]

            loss_real = real_loss_per_sample.mean()
            loss_synth = synth_loss_per_sample.mean()

            l_reg_r = compute_reg_batch(
                logit_scale=logit_scale, 
                samples=samples,
                feats=real_imgs_embedding, 
                labels=real_labels, 
                centroids=mu)
            l_reg_s = compute_reg_batch(
                logit_scale=logit_scale, 
                samples=samples,
                feats=synth_imgs_embedding, 
                labels=synth_labels, 
                centroids=mu)

            l_consistency = compute_consistency_reg(
                real_per_sample_loss=real_loss_per_sample,
                synth_per_sample_loss=synth_loss_per_sample,
                real_labels=real_labels,
                synth_labels=synth_labels,
                num_classes=mu.shape[0],
            )

            total_loss = (
                loss_real + lamda1 * loss_synth
                + lamda2 * l_reg_r + lamda3 * l_reg_s
                + lamda4 * l_consistency
            )

        scaler.scale(total_loss).backward()
        scaler.step(opt_h)
        scaler.update()

        writer.add_scalar("Loss_Batch/Real_Base", loss_real.item(), step)
        writer.add_scalar("Loss_Batch/Real_Reg", l_reg_r.item(), step)
        writer.add_scalar("Loss_Batch/Synth_Base", loss_synth.item(), step)
        writer.add_scalar("Loss_Batch/Synth_Reg", l_reg_s.item(), step)
        writer.add_scalar("Loss_Batch/Consistency", l_consistency.item(), step)
        writer.add_scalar("Loss_Batch/Total", total_loss.item(), step)

        del total_loss, logits_real_all, logits_synth_all

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
    prompt_path = "prompts.json"
    n_synth_per_class = 64
    lamda1 = 0.1
    lamda2 = 0.04
    lamda3 = 0.02
    lamda4 = 0.01

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

    exp_name = f"{dataset}_{n_samples_per_class}shot_{n_synth_per_class}synth_lamda{lamda1}_{lamda2}_{lamda3}_{lamda4}_spc"

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
            prompt_path=prompt_path,
            opt_h=opt_h, 
            scaler=scaler, 
            step=step, 
            fewshot_train_loader=fewshot_train_loader,
            synth_iter=synth_iter,
            lamda1=lamda1,
            lamda2=lamda2,
            lamda3=lamda3,
            lamda4=lamda4,
            lr_schedule=lr_schedule,
            writer=writer, 
            device=device, 
            dataset=dataset,
            logit_scale=logit_scale
        )

        model.model.eval()
        with torch.no_grad():
            mu_eval, _ = get_prototype(model, prompt_path)
            mu_eval = mu_eval.to(device)
            train_acc = get_acc(model, fewshot_train_loader, mu_eval, device)
            test_acc = get_acc(model, test_loader, mu_eval, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_path = os.path.join(ckpt_path, "best_spc_model")
            model.model.save_pretrained(best_model_path)

        writer.add_scalar("Epoch/Train_Accuracy", train_acc, global_step=epoch)
        writer.add_scalar("Epoch/Eval_Accuracy", test_acc, global_step=epoch)
        
        print(f"Epoch {epoch} | Train ACC: {train_acc*100:.2f}% | Test ACC: {test_acc*100:.2f}%")

        gc.collect()
    writer.close()
    
if __name__ == "__main__":
    main()