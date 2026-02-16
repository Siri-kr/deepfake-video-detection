import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("deepfake_model.pth"))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ["FAKE", "REAL"]

# Test folder
test_dir = "test_images"

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    print(f"{img_name} -> {classes[predicted.item()]}")
