from PIL import Image
import torch
from torchvision import transforms

# Transformation to match ViT input size and format
fingerprint_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

def preprocess_fingerprint_image(image_path):
    """
    Loads and preprocesses a fingerprint image for ViT model.
    """
    image = Image.open(image_path).convert("RGB")
    tensor = fingerprint_transform(image)
    return tensor  # Shape: [3, 224, 224]
