import os, time, torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoImageProcessor, AutoModelForImageClassification

MRL = "/home/bobby/eye_datasets/data"
path = "MichalMlodawski/open-closed-eye-classification-focalnet-base"
print("Loading FocalNet-base...")
processor = AutoImageProcessor.from_pretrained(path)
model = AutoModelForImageClassification.from_pretrained(path)
model.eval()
# LABEL_0=closed, LABEL_1=open

for split in ["test", "val"]:
    for sample_n in [500, 2000, 5000]:
        y_true, y_pred = [], []
        t0 = time.time()
        for label_idx, cls in enumerate(["awake", "sleepy"]):
            d = os.path.join(MRL, split, cls)
            fnames = sorted([f for f in os.listdir(d) if f.endswith(".png")])[:sample_n]
            for fname in fnames:
                img = Image.open(os.path.join(d, fname)).convert("RGB")
                inputs = processor(img, return_tensors="pt")
                with torch.no_grad():
                    out = model(**inputs)
                pred = 1 - out.logits.argmax(-1).item()  # 0=closed->1, 1=open->0
                y_true.append(label_idx)
                y_pred.append(pred)
        elapsed = time.time() - t0
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        print(f"{split} n={sample_n}: Acc={acc:.4f} F1={f1:.4f} P={prec:.4f} R={rec:.4f} CM={cm.tolist()} ({elapsed:.1f}s)")

print("DONE")
