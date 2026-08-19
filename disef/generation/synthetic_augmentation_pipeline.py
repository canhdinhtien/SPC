import argparse
import os
import sys
import pickle
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as tfms
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm
import numpy as np

from llava_processor import LlaVaProcessor
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data import get_data_loader
from utils import fix_random_seeds, get_dataset_name_for_template
from util_data import SUBSET_NAMES, TEMPLATES_SMALL
from models.qwen.src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from qwen_utils import get_image_embedding


def pad_image(image):
    width, height = image.size
    if width > height:
        new_width = width
        new_height = width
    else:
        new_width = height
        new_height = height
    new_im = Image.new("RGB", (new_width, new_height))
    new_im.paste(image, ((new_width - width) // 2, (new_height - height) // 2))
    return new_im.resize((512, 512))

def load_image(p):
    img = Image.open(p).convert("RGB")
    img = pad_image(img)
    return img

@torch.no_grad()
def pil_to_latents(image, vae):
    init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
    init_image = init_image.to(device="cuda", dtype=torch.float16)
    init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215
    return init_latent_dist

@torch.no_grad()
def path_to_latents(p, vae, mixup):
    if isinstance(p, list):
        images = []
        latents = []
        for p_ in p:
            image = load_image(p_)
            images.append(image)

        if mixup:
            lambdas = np.random.beta(0.2, 0.2, size=len(images))
            mixed_images = []
            for i in range(len(images)):
                idx1 = random.randint(0, len(images) - 1)
                idx2 = random.randint(0, len(images) - 1)
                mixed_image = Image.blend(images[idx1], images[idx2], lambdas[i])
                mixed_images.append(mixed_image)
            latents = [pil_to_latents(image, vae) for image in mixed_images]
        else:
            latents = [pil_to_latents(image, vae) for image in images]
        return torch.cat(latents)
    else:
        image = load_image(p)
        return pil_to_latents(image, vae)

@torch.no_grad()
def latents_to_pil(latents, vae):
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

@torch.no_grad()
def text_enc(prompts, tokenizer, text_encoder, maxlen=None):
    if maxlen is None:
        maxlen = tokenizer.model_max_length
    inp = tokenizer(
        prompts,
        padding="max_length",
        max_length=maxlen,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(inp.input_ids.to("cuda"))[0].half()

def load_llava(llava_model_path):
    print("Loading LLAVA")
    disable_torch_init()
    model_name = get_model_name_from_path(llava_model_path)
    (llava_tokenizer, llava_model, llava_image_processor, context_len) = load_pretrained_model(llava_model_path, None, model_name, False, False)
    llava_processor = LlaVaProcessor(llava_tokenizer, llava_image_processor, llava_model.config.mm_use_im_start_end)
    print("Loaded LLAVA")
    return llava_tokenizer, llava_model, llava_image_processor, context_len, llava_processor


sd_models = {
    "stable_diffusion": "sd2-community/stable-diffusion-2-1",
    "realistic": "SG161222/Realistic_Vision_V2.0",
}

def load_stable_diffusion(model_name):
    print("Loading Stable Diffusion Pipeline")
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_models[model_name], torch_dtype=torch.float16, local_files_only=True
    )
    pipe = pipe.to("cuda")
    scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(20)
    pipe.scheduler = scheduler
    print("Loaded Stable Diffusion Pipeline")
    return pipe.vae, pipe.unet, pipe.scheduler, pipe.tokenizer, pipe.text_encoder

def caption_image(images_path, class_name, llava_tokenizer, llava_model, llava_processor):
    query = f"Provide a detailed caption for the image focusing on the object, knowing it's a {class_name}."
    batch_size = len(images_path)
    batch_images, batch_text = llava_processor.get_processed_tokens_batch([query] * batch_size, images_path)

    conv = conv_templates[llava_processor.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    input_ids = batch_text.cuda()
    image_tensor = batch_images

    with torch.inference_mode():
        output_ids = llava_model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
        )

    generated_outputs = llava_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
    generated_outputs = [out.strip() for out in generated_outputs]
    generated_outputs = [out[:-len(stop_str)] if out.endswith(stop_str) else out for out in generated_outputs]
    return [out.strip() for out in generated_outputs]


@torch.no_grad()
def get_prototype(model, dataset):
    class_names = SUBSET_NAMES[dataset]
    dataset_name = get_dataset_name_for_template(dataset)
    all_mus = []
    
    if hasattr(model, 'model'):
        model.model.eval()

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

def load_qwen_model(qwen_model_path):
    print("Loading Qwen3-VL...")
    model = Qwen3VLEmbedder(model_name_or_path=qwen_model_path)
    model.model.eval()
    if next(model.model.parameters()).device == torch.device('cpu'):
         model.model = model.model.to("cuda")
    print("Loaded Qwen3-VL.")
    return model

def get_filtered_paths(dataset_name, n_samples_per_class, real_data_dir):
    print(f"Lọc {n_samples_per_class}-shot dữ liệu gốc từ {real_data_dir}...")
    train_loader, _ = get_data_loader(
        real_train_data_dir=real_data_dir,
        real_test_data_dir=real_data_dir,
        dataset=dataset_name,
        bs=1, eval_bs=1,
        n_img_per_cls=n_samples_per_class,
        model_type="qwen"
    )
    
    train_dataset = train_loader.dataset
    class_names = SUBSET_NAMES[dataset_name]
    paths_dict = {cls: [] for cls in class_names}
    
    if dataset_name in ['dtd', 'flowers102', 'food101', 'sun397', 'fgvc_aircraft']:
        _images = train_dataset._image_files
        _labels = train_dataset._labels
    elif dataset_name == 'eurosat':
        _images = [sample[0] for sample in train_dataset.samples]
        _labels = [sample[1] for sample in train_dataset.samples]
    elif dataset_name == 'pets':
        _images = train_dataset._images
        _labels = train_dataset._labels
    elif dataset_name == 'cars':
        _images = [sample[0] for sample in train_dataset._samples]
        _labels = [sample[1] for sample in train_dataset._samples]
    elif dataset_name == 'caltech101':
        _images = [
            os.path.join(
                train_dataset.root, 
                "caltech101", 
                "101_ObjectCategories", 
                train_dataset.categories[train_dataset.y[i]], 
                f"image_{train_dataset.index[i]:04d}.jpg"
            ) for i in range(len(train_dataset.index))
        ]
        _labels = train_dataset.y
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    for img_path, label in zip(_images, _labels):
        class_name = class_names[label]
        paths_dict[class_name].append(img_path)
        
    return paths_dict, class_names


def main():
    fix_random_seeds(22)

    dataset = "dtd"
    starting_step = 5
    batch_size = 8
    cfg_strength = 8
    images_per_class = 64
    n_samples = 16
    real_data_dir = 16
    use_llava = True
    llava_model_path = "llava"
    sd_model = "sd2-community/stable-diffusion-2-1"
    output_dir = "synthetic_data"
    mixup = False

    paths_dict, class_names = get_filtered_paths(dataset, n_samples, real_data_dir)

    if dataset == "imagenet":
        inmap = {}
        with open("imagenet_map.txt", "r") as f:
            for line in f:
                line = line.strip().split(" ")
                inmap[line[0]] = line[2]
        class_names = [inmap[_] for _ in class_names]

        paths_dict = {new_k: paths_dict[old_k] for old_k, new_k in zip(list(paths_dict.keys()), class_names)}

    qwen_model = load_qwen_model("./models/Qwen3-VL-Embedding-2B")
    vae, unet, scheduler, tokenizer, text_encoder = load_stable_diffusion(sd_model)

    if use_llava:
        llava_tokenizer, llava_model, llava_image_processor, context_len, llava_processor = load_llava(llava_model_path)
    else:
        available_captions = pickle.load(open(f"captions/{dataset}.pkl", "rb"))

    zeroshot_weights = get_prototype(qwen_model, dataset).to("cuda")

    out_root = f"{output_dir}/{dataset}_{n_samples}shot"
    os.makedirs(out_root, exist_ok=True)
    print(f"Lưu ảnh tại: {out_root}")

    for idx, class_name in enumerate(tqdm(class_names)):
        output_dir = os.path.join(out_root, class_name)
        os.makedirs(output_dir, exist_ok=True)

        available_files = paths_dict[class_name]
        generated_this_class = len(os.listdir(output_dir))
        cycles_spent_this_class = 0
        prompt_cache = {}

        while generated_this_class < images_per_class:
            cycles_spent_this_class += 1
            
            file_paths = random.choices(available_files, k=batch_size)

            latents = path_to_latents(file_paths, vae, mixup)
            noise = torch.randn_like(latents)
            noised_latents = scheduler.add_noise(latents, noise, timesteps=torch.tensor([scheduler.timesteps[starting_step]]))

            prompts = []
            if use_llava:
                missing_file_paths = [p for p in file_paths if p not in prompt_cache]

                if len(missing_file_paths) > 0:
                    missing_prompts = caption_image(missing_file_paths, class_name, llava_tokenizer, llava_model, llava_processor)
                    for i, p in enumerate(missing_file_paths):
                        prompt_cache[p] = missing_prompts[i]

                for p in file_paths:
                    prompts.append(prompt_cache[p])
                prompts = random.sample(prompts, len(prompts))
            else:
                prompts_this_class = available_captions[class_name]
                try:
                    prompts = random.choices(prompts_this_class, k=batch_size)
                except:
                    prompts = [f"a photo of a {class_name}."] * batch_size

            text_embed = torch.cat([text_enc([prompt], tokenizer, text_encoder) for prompt in prompts])
            uncond = text_enc([""] * 1, tokenizer, text_encoder, text_embed.shape[1])
            uncond = uncond.repeat(batch_size, 1, 1)
            emb = torch.cat([uncond, text_embed])

            latents = noised_latents

            for i, ts in enumerate(tqdm(scheduler.timesteps[starting_step:], disable=True)):
                inp = scheduler.scale_model_input(torch.cat([latents] * 2), ts)
                with torch.no_grad(), torch.autocast("cuda"):
                    unconditional, conditional = unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)
                predicted_sample = unconditional + cfg_strength * (conditional - unconditional)
                latents = scheduler.step(predicted_sample, ts, latents).prev_sample

            final_imgs = latents_to_pil(latents, vae)
            final_imgs_highres = [img.resize((512, 512)) for img in final_imgs]
            
            final_imgs_qwen = [img.resize((256, 256)) for img in final_imgs]

            with torch.no_grad(), torch.amp.autocast("cuda"):
                image_features = get_image_embedding(qwen_model, final_imgs_qwen)
                image_features = F.normalize(image_features, dim=-1)

            similarities = (image_features @ zeroshot_weights.T).softmax(dim=-1)

            for i in range(batch_size):
                predicted_class = similarities[i].argmax().item()

                if predicted_class == idx or (cycles_spent_this_class > 100):
                    random_name = random.randint(0, 1000000000)
                    out_path = os.path.join(output_dir, f"{class_name}_{generated_this_class}_{random_name}.jpg")
                    final_imgs_highres[i].save(out_path)
                    generated_this_class = len(os.listdir(output_dir))


if __name__ == "__main__":
    main()