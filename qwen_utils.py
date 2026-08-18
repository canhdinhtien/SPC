import torch
import torch.nn.functional as F

from tqdm import tqdm

def get_image_embedding(model, images):
    image_inputs = [{"image": img} for img in images]
    image_embs = model.process(image_inputs)
    image_embs = F.normalize(image_embs, dim=-1)
    return image_embs

def get_acc(model, data_loader, mu, device):
    model.model.to(device)
    mu.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            labels = labels.to(device)
            image_embedding = get_image_embedding(model, images)
            # compute similarity and predict
            similarity = image_embedding @ mu.T
            preds = similarity.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total