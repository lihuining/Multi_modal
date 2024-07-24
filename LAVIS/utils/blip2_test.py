from lavis.models import load_model
import torch
from PIL import Image
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raw_image = Image.open("/media/allenyljiang/564AFA804AFA5BE5/Codes/STG-NF/utils/10_0075_grid_plot.png").convert("RGB")
print(raw_image.size)
raw_image = raw_image.resize((224,224))
model = load_model("blip2", "pretrain")

loader = transforms.Compose([
    transforms.ToTensor()])
image_tensor = loader(raw_image).unsqueeze(0)
query_output,image_embeds = model.forward_image(image_tensor) # 输入必须是224×224，[1,32,768],[1,257,1408]
print(query_output.shape)
print(image_embeds.shape)
# features_image = model.extract_features(sample, mode="image")
# features_text = model.extract_features(sample, mode="text")
# print(features_image.image_embeds.shape)
# # torch.Size([1, 197, 768])
# print(features_text.text_embeds.shape)
# # torch.Size([1, 12, 768])
#
# # low-dimensional projected features
# print(features_image.image_embeds_proj.shape)
# # torch.Size([1, 197, 256])
# print(features_text.text_embeds_proj.shape)
# # torch.Size([1, 12, 256])
# similarity = features_image.image_embeds_proj[:,0,:] @ features_text.text_embeds_proj[:,0,:].t()
# print(similarity)
# # tensor([[0.2622]])