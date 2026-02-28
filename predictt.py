import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
import os

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Load Model Architecture
# ----------------------------
model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
model.head = nn.Linear(model.head.in_features, 7)
model = model.to(device)

# ----------------------------
# Load Saved Weights
# ----------------------------
model_path = "vit_7class_model.pth"

if not os.path.exists(model_path):
    print("Model file not found!")
    exit()

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Model Loaded Successfully\n")

# ----------------------------
# Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# ----------------------------
# Image Input
# ----------------------------
image_path = input("Enter image filename: ")

if not os.path.exists(image_path):
    print("Image not found!")
    exit()

img = Image.open(image_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

# ----------------------------
# Prediction
# ----------------------------
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output,1)

classes = ['Normal','mel','bkl','bcc','akiec','vasc','df']

idx = torch.argmax(probs).item()
confidence = probs[0][idx].item()

print("\n===== RESULT =====")
print("Prediction :", classes[idx])
print("Confidence :", round(confidence*100,2), "%")

if idx == 0:
    print("Final Result : NORMAL")
else:
    print("Final Result : ABNORMAL")
    print("Lesion Type  :", classes[idx])