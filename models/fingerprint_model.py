import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import joblib  # for label encoder persistence

class FingerprintViT(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(FingerprintViT, self).__init__()
        config = ViTConfig()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224") if pretrained else ViTModel(config)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)

def load_fingerprint_model(device):
    # Load shared label encoder
    label_encoder = joblib.load("weights/shared_label_encoder.pkl")
    num_classes = len(label_encoder.classes_)

    # Initialize model with correct output dimension
    model = FingerprintViT(num_classes=num_classes)
    model.load_state_dict(torch.load("weights/best_fingerprint_vit.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, label_encoder

def preprocess_fingerprint_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)
