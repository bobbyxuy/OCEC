#!/usr/bin/env python3
"""Train eye open/closed classifier on MRL using DINOv2-ViT-Large.
Target: F1 > 0.99
"""
import os, time, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import timm

DATA_DIR = "/home/bobby/eye_datasets/data"
IMG_SIZE = 64
BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "vit_large_patch14_dinov2.lvd142m"

class EyeDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        for label, cls in enumerate(["awake", "sleepy"]):
            d = os.path.join(root, split, cls)
            for f in os.listdir(d):
                if f.endswith(".png"):
                    self.samples.append((os.path.join(d, f), label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L").convert("RGB")  # grayscale -> 3ch
        if self.transform:
            img = self.transform(img)
        return img, label

def get_transforms(mode):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading DINOv2-ViT-Large backbone...")
    backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=IMG_SIZE)
    feat_dim = backbone.embed_dim  # 1024
    
    model = nn.Sequential(
        backbone,
        nn.LayerNorm(feat_dim),
        nn.Linear(feat_dim, 2),
    ).to(DEVICE)
    
    # Phase 1: freeze backbone
    for p in model[0].parameters():
        p.requires_grad = False
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total_params/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")

    train_ds = EyeDataset(DATA_DIR, "train", get_transforms("train"))
    val_ds = EyeDataset(DATA_DIR, "val", get_transforms("val"))
    test_ds = EyeDataset(DATA_DIR, "test", get_transforms("val"))
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0
    for epoch in range(EPOCHS):
        if epoch == 5:
            for p in model[0].parameters():
                p.requires_grad = True
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"E{epoch}: Unfroze backbone. Trainable: {trainable/1e6:.1f}M")
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR*0.1, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-epoch)

        model.train()
        running_loss = 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        elapsed = time.time() - t0
        
        val_acc, val_f1, val_cm = evaluate(model, val_loader, DEVICE)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_dinov2_eye.pth")
        print(f"E{epoch+1:2d} loss={running_loss/len(train_loader):.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} best={best_f1:.4f} ({elapsed:.0f}s)")
        if val_f1 >= 0.999:
            print("F1 >= 0.999, early stop!")
            break

    model.load_state_dict(torch.load("best_dinov2_eye.pth"))
    test_acc, test_f1, test_cm = evaluate(model, test_loader, DEVICE)
    print(f"\n=== TEST === Acc={test_acc:.4f} F1={test_f1:.4f} CM={test_cm.tolist()}")
    
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    torch.onnx.export(model, dummy, "dinov2_eye_classifier.onnx",
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}})
    os.system("ls -lh dinov2_eye_classifier.onnx")

if __name__ == "__main__":
    main()
