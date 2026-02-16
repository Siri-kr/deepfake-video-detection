import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# paths
DATA_DIR = "faces"
MODEL_SAVE_PATH = "deepfake_model.pth"

# settings
BATCH_SIZE = 16
EPOCHS = 5
IMG_SIZE = 224

# transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# load pretrained model
model = models.resnet18(pretrained=True)

# change final layer (2 classes)
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print("🚀 Training started...")

for epoch in range(EPOCHS):
    running_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss:.4f}")

# save model
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("✅ Training completed & model saved!")
