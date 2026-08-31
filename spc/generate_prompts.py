import os
import re
import json
import random
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from data import get_data_loader
from util_data import SUBSET_NAMES
from models.qwen.src.models.qwen3_vl_embedding import Qwen3VLEmbedder

from utils import fix_random_seeds

def setup_models(embedder_path="./models/Qwen3-VL-Embedding-2B", qwen_vl_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    embedder = Qwen3VLEmbedder(model_name_or_path=embedder_path)
    embedder.model.eval()
    embedder.model.to("cuda")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    qwen_vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_vl_name, quantization_config=bnb_config, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(qwen_vl_name)
    
    return embedder, qwen_vl, processor

def build_fewshot_dict_from_loader(data_dir, dataset_name, class_names, n_shot=16, batch_size=32):
    train_loader, _ = get_data_loader(
        real_train_data_dir=data_dir,
        dataset=dataset_name,
        bs=batch_size,
        eval_bs=batch_size,
        is_rand_aug=False,
        n_img_per_cls=n_shot,
        model_type="qwen"
    )

    fewshot_dict = {cls: [] for cls in class_names}
    
    for batch in tqdm(train_loader, desc="Extracting PIL Images from DataLoader"):
        images, labels = batch[0], batch[1]
        for img, label_idx in zip(images, labels):
            fewshot_dict[class_names[label_idx.item()]].append(img)
            
    return fewshot_dict

def sample_images_from_dict(fewshot_dict, class_name, k=4):
    images = fewshot_dict.get(class_name, [])
    return random.sample(images, k) if len(images) >= k else images

def normalize_class_name(name):
    name = str(name).lower().strip()
    return re.sub(r"\s+", " ", name).strip(": ")

def clean_description(text):
    text = str(text)
    text = re.sub(r'\*{1,3}', '', text).strip("'\"")
    text = re.sub(r',\s*$', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def parse_class_prompts(text):
    results = {}
    header_pattern = re.compile(r"(?:\*\*)?(?:Refined Prompts for )?Class:\s*([^\n*]+)(?:\*\*)?", re.I)
    headers = list(header_pattern.finditer(text))

    for i, h in enumerate(headers):
        class_name = h.group(1).strip().lower()
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[start:end]

        prompts = []
        for m in re.finditer(r"^\d+\.\s*(.+)$", block, re.M):
            prompts.append(clean_description(m.group(1)))
        results[class_name] = prompts
    return results

def match_class_name(target_class, parsed_dict):
    target = normalize_class_name(target_class)
    for key in parsed_dict:
        if normalize_class_name(key) == target:
            return key
    candidates = [k for k in parsed_dict if target in normalize_class_name(k) or normalize_class_name(k) in target]
    return candidates[0] if len(candidates) == 1 else None

@torch.no_grad()
def compute_balanced_lda_loss(embedder, prompts_by_class, eps=1e-8):
    class_embeddings, class_means = [], []
    for prompts in prompts_by_class:
        z = embedder.process([{"text": s} for s in prompts]).to("cuda")
        class_embeddings.append(z)
        class_means.append(z.mean(dim=0))
        
    class_means = torch.stack(class_means)
    global_mean = class_means.mean(dim=0)
    
    within_terms = [(z - mu_c.unsqueeze(0)).pow(2).sum(dim=1).mean() for z, mu_c in zip(class_embeddings, class_means)]
    tr_SW = torch.stack(within_terms).mean()
    tr_SB = ((class_means - global_mean.unsqueeze(0)).pow(2).sum(dim=1)).mean()
    
    lda_ratio = tr_SB / (tr_SW + eps)
    lda_loss = -torch.log(lda_ratio + eps)
    return lda_loss, tr_SB, tr_SW

@torch.no_grad()
def find_topk_hard_lda_pairs(embedder, prompts_by_class, k=10, eps=1e-8):
    C = len(prompts_by_class)
    class_means, class_vars = [], []

    for prompts in prompts_by_class:
        z = embedder.process([{"text": s} for s in prompts]).to("cuda")
        mu = z.mean(dim=0)
        var = ((z - mu.unsqueeze(0)).pow(2).sum(dim=1)).mean()
        class_means.append(mu)
        class_vars.append(var)

    class_means = torch.stack(class_means)
    class_vars = torch.stack(class_vars)
    pairs = []

    for i in range(C):
        for j in range(i + 1, C):
            sb = (class_means[i] - class_means[j]).pow(2).sum()
            sw = 0.5 * (class_vars[i] + class_vars[j])
            lda_ratio = sb / (sw + eps)
            lda_loss = -torch.log(lda_ratio + eps)
            pairs.append({"class_i": i, "class_j": j, "lda_loss": float(lda_loss), "ratio": float(lda_ratio)})

    pairs.sort(key=lambda x: x["lda_loss"], reverse=True)
    
    selected, used = [], set()
    for p in pairs:
        i, j = p["class_i"], p["class_j"]
        if i in used or j in used: 
            continue
        selected.append(p)
        used.add(i)
        used.add(j)
        if len(selected) == k:
            break
    return selected

@torch.no_grad()
def filter_pairwise_prompts(embedder, prompts_A, prompts_B, num_keep=30):
    zA = F.normalize(embedder.process([{"text": p} for p in prompts_A]).to("cuda"), dim=-1)
    zB = F.normalize(embedder.process([{"text": p} for p in prompts_B]).to("cuda"), dim=-1)

    mu_A = F.normalize(zA.mean(dim=0, keepdim=True), dim=-1)
    mu_B = F.normalize(zB.mean(dim=0, keepdim=True), dim=-1)

    score_A = F.cosine_similarity(zA, mu_A, dim=-1) - F.cosine_similarity(zA, mu_B, dim=-1)
    score_B = F.cosine_similarity(zB, mu_B, dim=-1) - F.cosine_similarity(zB, mu_A, dim=-1)

    idxA = torch.topk(score_A, min(num_keep, len(prompts_A))).indices.cpu().tolist()
    idxB = torch.topk(score_B, min(num_keep, len(prompts_B))).indices.cpu().tolist()

    return [prompts_A[i] for i in idxA], [prompts_B[i] for i in idxB]

def generate_initial_prompts(qwen_model, processor, class_name, target_num=30):
    instruction = f"""
You are an expert in visual recognition.
Generate exactly {target_num} distinct visual descriptions for the class "{class_name}".
Focus on concrete visual features (shape, color, texture, parts, structure).
Do not use generic wording, definitions, or non-visual technical specs.

Output exactly {target_num} descriptions. Format strictly:
1. ...
2. ...
"""
    content = [{"type": "text", "text": instruction}]
    text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(qwen_model.device)
    
    generated_ids = qwen_model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.8, top_p=0.95)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    
    prompts = []

    for m in re.finditer(r"^\d+\.\s*(.+)$", output_text, re.M):
        prompts.append(clean_description(m.group(1)))
        
    return prompts

def refine_prompts_by_lda_pair(qwen_model, processor, class_names, pre_prompts, class_a_idx, class_b_idx, class_a_images, class_b_images):
    name_A, name_B = class_names[class_a_idx], class_names[class_b_idx]
    prompts_A, prompts_B = pre_prompts[class_a_idx], pre_prompts[class_b_idx]
    
    instruction = f"""
You are refining prompts for a vision-language model.
You will be given few-shot images and current prompts for two visually confusing classes.

CLASS A: {name_A}
Current prompts:
{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(prompts_A[:10]))}

CLASS B: {name_B}
Current prompts:
{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(prompts_B[:10]))}

Task: Rewrite the prompts so that the two classes are more semantically separated. 
Focus STRONGLY on subtle discriminative differences (shape, parts, textures) from the provided images.
Avoid generic descriptions. 

Output exactly 30 refined prompts per class in English. Format strictly:
Class: {name_A}
1. ...
2. ...

Class: {name_B}
1. ...
2. ...
"""
    content = [{"type": "text", "text": instruction}]
    
    content.append({"type": "text", "text": f"Images of class: {name_A}"})
    for img in class_a_images: 
        content.append({"type": "image", "image": img})
        
    content.append({"type": "text", "text": f"Images of class: {name_B}"})
    for img in class_b_images: 
        content.append({"type": "image", "image": img})

    text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=class_a_images + class_b_images, padding=True, return_tensors="pt").to(qwen_model.device)
    
    generated_ids = qwen_model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.8, top_p=0.95, repetition_penalty=1.15)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

def main():
    seed = 21
    dataset = "dtd"
    data_dir = "train"
    output_dir = "./generated_prompts"
    n_runs = 5
    k_pairs = 10
    n_shot = 16
    k_images = 4
    target_num = 30

    fix_random_seeds(seed)
    os.makedirs(output_dir, exist_ok=True)
    classes = SUBSET_NAMES[dataset]
    
    embedder, qwen_vl, processor = setup_models()

    fewshot_dict = build_fewshot_dict_from_loader(
        data_dir=data_dir, dataset_name=dataset, class_names=classes, n_shot=n_shot
    )

    best_prompts = []
    
    for cls in tqdm(classes, desc="Generating Initial Prompts"):
        prompts_for_cls = []
        retry = 0
        
        while len(prompts_for_cls) < target_num and retry < 3:
            needed = target_num - len(prompts_for_cls)
            new_prompts = generate_initial_prompts(qwen_vl, processor, cls, target_num=needed)
            prompts_for_cls.extend(new_prompts)

            prompts_for_cls = list(dict.fromkeys(prompts_for_cls))
            retry += 1
            
        if len(prompts_for_cls) == 0:
            prompts_for_cls = [f"A photo showing visual details of {cls}."] * target_num
        elif len(prompts_for_cls) < target_num:
            repeats = target_num // len(prompts_for_cls) + 1
            prompts_for_cls = (prompts_for_cls * repeats)[:target_num]

        if len(prompts_for_cls) > target_num:
            prompts_for_cls = random.sample(prompts_for_cls, target_num)
            
        best_prompts.append(prompts_for_cls)
        torch.cuda.empty_cache()

    for run in range(n_runs):
        hard_pairs = find_topk_hard_lda_pairs(embedder, best_prompts, k=k_pairs)
        refined_prompts_dict = {key: [] for key in classes}
        
        for p in tqdm(hard_pairs, desc=f"Refining Top {k_pairs} Hard Pairs"):
            idx_i, idx_j = p['class_i'], p['class_j']
            class_i, class_j = classes[idx_i], classes[idx_j]
            
            imgs_a = sample_images_from_dict(fewshot_dict, class_i, k=k_images)
            imgs_b = sample_images_from_dict(fewshot_dict, class_j, k=k_images)
            
            generated = refine_prompts_by_lda_pair(
                qwen_vl, processor, classes, best_prompts, idx_i, idx_j, imgs_a, imgs_b
            )

            parsed = parse_class_prompts(generated)
            key_i = match_class_name(class_i, parsed)
            key_j = match_class_name(class_j, parsed)
            
            if key_i: 
                refined_prompts_dict[class_i] = parsed[key_i]
            if key_j: 
                refined_prompts_dict[class_j] = parsed[key_j]
                
            kept_A, kept_B = filter_pairwise_prompts(
                embedder,
                best_prompts[idx_i] + refined_prompts_dict.get(class_i, []),
                best_prompts[idx_j] + refined_prompts_dict.get(class_j, []),
                num_keep=target_num
            )
            
            best_prompts[idx_i] = kept_A
            best_prompts[idx_j] = kept_B
            
            torch.cuda.empty_cache() 
            
        _, _, _ = compute_balanced_lda_loss(embedder, best_prompts)

    final_dict = {classes[i]: best_prompts[i] for i in range(len(classes))}
    output_path = os.path.join(output_dir, f"refined_prompts_{dataset}_seed_{seed}.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dict, f, indent=4, ensure_ascii=False)
        
    print(f"\nFile Prompts is saved at: {output_path}")

if __name__ == "__main__":
    main()