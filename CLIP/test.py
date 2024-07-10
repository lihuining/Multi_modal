import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("/mnt/workspace/workgroup_share/lhn/multi_modal/CLIP/CLIP.png")).unsqueeze(0).to(device) # [1,3,224,224]
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device) # [3,77]

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy() # [1,3]

print("Label probs:", probs)  #