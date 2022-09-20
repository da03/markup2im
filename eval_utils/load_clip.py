import torch
import clip

from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def calc_clip_logits(pred_img: Image, gold_img: Image) -> float:
    pred_img, gold_img = preprocess(pred_img).to(device), preprocess(gold_img).to(device)

    with torch.no_grad():
        pred_image_features = model.encode_image(pred_img.unsqueeze(0))
        gold_image_features = model.encode_image(gold_img.unsqueeze(0))
        pred_image_features /= pred_image_features.norm(dim=-1, keepdim=True)
        gold_image_features /= gold_image_features.norm(dim=-1, keepdim=True)
        similarity = pred_image_features @ gold_image_features.T
    
    return similarity.item()

