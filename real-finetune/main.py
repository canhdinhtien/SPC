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

from data import get_data_loader
from utils import fix_random_seeds
from models.qwen.src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from qwen_utils import get_image_embedding, get_acc
from util_data import SUBSET_NAMES

from templates import TEMPLATES

def get_prototype(model, dataset):
    templates = TEMPLATES[dataset]
    class_names = SUBSET_NAMES[dataset]

    all_mus = []

    for class_name in class_names:
        class_texts = [
            {"text": template.format(class_name)}
            for template in templates
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
    writer,
    device,
    dataset="dtd",
    logit_scale=15,
):
    model.model.train()

    for real_images, real_labels in fewshot_train_loader:
        step += 1
        real_labels = real_labels.to(device)

        opt_h.zero_grad(set_to_none=True)

        mu = get_prototype(model, dataset)

        with torch.amp.autocast("cuda"):
            real_imgs_embedding = get_image_embedding(model, real_images)
            logits_real_all = logit_scale * (real_imgs_embedding @ mu.t())

            loss_real = F.cross_entropy(logits_real_all, real_labels)

        loss_value = loss_real.item()

        scaler.scale(loss_real).backward()
        scaler.step(opt_h)
        scaler.update()

        writer.add_scalar("Loss", loss_value, step)

        del real_images, real_labels, real_imgs_embedding, logits_real_all, loss_real, mu

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

    fewshot_train_loader, test_loader = get_data_loader(
        real_train_data_dir="",
        real_test_data_dir="",
        dataset=dataset,
        bs=batch_size,
        eval_bs=eval_batch_size,
        n_img_per_cls=n_samples_per_class,
        model_type=model_type
    )

    exp_name = f"{dataset}_qwen_lora_{n_samples_per_class}shot"

    print(f"Number of few-shot training samples: {len(fewshot_train_loader.dataset)}")
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

    step = 0
    best_test_acc = 0.0

    for epoch in range(n_epochs):
        step = train_one_epoch(
            model=model, 
            opt_h=opt_h, 
            scaler=scaler, 
            step=step, 
            fewshot_train_loader=fewshot_train_loader, 
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
            best_model_path = os.path.join(ckpt_path, "best_real_finetuned_model")
            model.model.save_pretrained(best_model_path)

        writer.add_scalar("Epoch/Train_Accuracy", train_acc, global_step=epoch)
        writer.add_scalar("Epoch/Eval_Accuracy", test_acc, global_step=epoch)
        
        print(f"Epoch {epoch} | Train ACC: {train_acc*100:.2f}% | Test ACC: {test_acc*100:.2f}%")

        gc.collect()
    writer.close()
    
if __name__ == "__main__":
    main()