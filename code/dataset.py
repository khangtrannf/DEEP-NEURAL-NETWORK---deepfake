# dataset.py
import warnings
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import kagglehub

def get_dataloaders(batch_size=32, seed=42):
    """
    Tải dataset, thực hiện Augmentation (Train) & Normalization (All),
    sau đó chia tập Train/Val/Test theo tỉ lệ 80/10/10.
    """
    # 1. Download dataset
    dataset_path = kagglehub.dataset_download("saurabhbagchi/deepfake-image-detection")
    
    # 2. Pipeline Augmentation & Normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load & Split (80/10/10)
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)
    
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform

    # 4. DataLoaders
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_pin_memory)

    return train_loader, val_loader, test_loader, full_dataset.classes

if __name__ == "__main__":
    # Test thử khi chạy trực tiếp file dataset.py
    print("--- Kiểm tra Pipeline Dataset ---")
    train_loader, val_loader, test_loader, classes = get_dataloaders(batch_size=32)
    print(f"-> Danh sách lớp: {classes}")
    print(f"-> Só lượng batch Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")
    
    # Lay 1 batch kiem tra shape
    images, labels = next(iter(train_loader))
    print(f"-> Image Batch Shape: {images.shape} (Batch_size, Channels, Height, Width)")
    print(f"-> Label Batch Shape: {labels.shape}")
    print(" Pipeline hoạt động hoàn hảo!")