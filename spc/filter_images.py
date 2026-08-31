import os
import shutil
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from peft import PeftModel

from data import DatasetSynthImage, get_transforms, raw_image_collate_fn
from models.qwen.src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from qwen_utils import get_image_embedding
from util_data import SUBSET_NAMES, TEMPLATES_SMALL
from utils import get_dataset_name_for_template

class DatasetSynthImageWithPaths(DatasetSynthImage):
    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        image_path = self.image_paths[idx]

        return out[0], out[1], image_path

def filter_collate_fn(batch):
    paths = [item[2] for item in batch]
    orig_batch = [(item[0], item[1]) for item in batch]
    collated_images, collated_labels = raw_image_collate_fn(orig_batch)

    return collated_images, collated_labels, paths


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

def classify_generated_images_by_confidence(
    model, data_loader, mu, logit_scale, device, output_dir, class_names
):
    model.model.eval()
    mu = mu.to(device)
    results = []

    print("\n Computing Confidence")
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for images, labels, paths in tqdm(data_loader, desc="Compute confidence"):
            labels = labels.to(device)
            image_embedding = get_image_embedding(model, images)
            logits = logit_scale * (image_embedding @ mu.T)
            probs = F.softmax(logits, dim=1)
            pred_conf, pred_labels = probs.max(dim=1)

            for label, pred_label, conf, path in zip(labels.cpu(), pred_labels.cpu(), pred_conf.cpu(), paths):
                results.append({
                    "path": str(path),
                    "label": int(label.item()),
                    "predicted_class": int(pred_label.item()),
                    "confidence": float(conf.item()),
                })

    result_df = pd.DataFrame(results)

    for region in ["Low", "Medium", "High"]:
        os.makedirs(os.path.join(output_dir, region), exist_ok=True)

    thresholds = {}
    for class_id, group in result_df.groupby("label"):
        group = group.sort_values("confidence", ascending=True)
        n = len(group)
        confidence_values = group["confidence"].to_numpy()

        q33 = float(pd.Series(confidence_values).quantile(1 / 3))
        q67 = float(pd.Series(confidence_values).quantile(2 / 3))
        thresholds[class_id] = {"q33": q33, "q67": q67, "n": n}

        low_end = n // 3
        medium_end = 2 * n // 3
        sorted_indices = group.index.tolist()

        for rank, idx in enumerate(sorted_indices):
            if rank < low_end:
                region = "Low"
            elif rank < medium_end:
                region = "Medium"
            else:
                region = "High"
            result_df.loc[idx, "region"] = region

    for _, row in tqdm(result_df.iterrows(), total=len(result_df), desc="Copy images"):
        class_id = int(row["label"])
        region = row["region"]
        path = row["path"]

        class_dir = os.path.join(output_dir, region, class_names[class_id])
        os.makedirs(class_dir, exist_ok=True)

        filename = os.path.basename(path)
        output_path = os.path.join(class_dir, filename)

        if os.path.exists(output_path):
            name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(output_path):
                new_filename = f"{name}_{counter}{ext}"
                output_path = os.path.join(class_dir, new_filename)
                counter += 1

        shutil.copy2(path, output_path)

    # counts = result_df.groupby(["label", "region"]).size().unstack(fill_value=0).reindex(columns=["Low", "Medium", "High"], fill_value=0)

    return result_df, thresholds

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "dtd"
    model_type = "qwen"
    logit_scale = 15
    batch_size = 32

    base_model_path = "./models/Qwen3-VL-Embedding-2B"
    lora_weights_path = "best_real_finetuned_model"
    synth_train_data_dir = "dtd_synthetic_data"
    output_image_dir = "./synthetic_regions"

    print("Loading Base Model Qwen...")
    model = Qwen3VLEmbedder(model_name_or_path=base_model_path)

    print(f"Loading LoRA weights from: {lora_weights_path}...")
    model.model = PeftModel.from_pretrained(model.model, lora_weights_path)
    model.model.to(device)
    model.model.eval()

    _, test_transform = get_transforms(model_type)

    filter_dataset = DatasetSynthImageWithPaths(
        synth_train_data_dir=synth_train_data_dir,
        transform=test_transform,
        dataset=dataset,
        n_img_per_cls=192,
        is_pooled_fewshot=False
    )

    data_loader = DataLoader(
        filter_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=filter_collate_fn
    )

    with torch.no_grad():
        mu_eval = get_prototype(model, dataset)

    class_names = SUBSET_NAMES[dataset]

    print("\nFiltering...")
    _, _ = classify_generated_images_by_confidence(
        model=model,
        data_loader=data_loader,
        mu=mu_eval,
        logit_scale=logit_scale,
        device=device,
        output_dir=output_image_dir,
        class_names=class_names
    )

    print(f"\nImages were saved at: {output_image_dir}")

if __name__ == "__main__":
    main()