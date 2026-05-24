#!/usr/bin/env python3
import os, time, numpy as np, cv2, torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoImageProcessor, AutoModelForImageClassification

MRL_BASE = '/home/bobby/eye_datasets/data'

MODELS = [
    ('FocalNet-base', 'MichalMlodawski/open-closed-eye-classification-focalnet-base'),
    ('MobileNetV2', 'MichalMlodawski/open-closed-eye-classification-mobilev2'),
]

def eval_on_mrl(model_name, model_path, split='test'):
    print(f'\n===== {model_name} on MRL {split} =====')
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.eval()
    
    # Check label mapping
    id2label = model.config.id2label
    print(f'  Labels: {id2label}')
    
    y_true, y_pred = [], []
    t0 = time.time()
    
    for label_idx, cls_name in enumerate(['awake', 'sleepy']):
        d = os.path.join(MRL_BASE, split, cls_name)
        fnames = [f for f in os.listdir(d) if f.endswith('.png')][:2000]  # sample 2000 per class for speed
        
        for fname in fnames:
            img = Image.open(os.path.join(d, fname)).convert('RGB')
            inputs = processor(img, return_tensors='pt')
            with torch.no_grad():
                out = model(**inputs)
            pred_id = out.logits.argmax(-1).item()
            pred_label = id2label[pred_id].lower()
            
            # Map pred to 0=open, 1=closed
            pred = 0 if 'open' in pred_label else 1
            y_true.append(label_idx)
            y_pred.append(pred)
    
    elapsed = time.time() - t0
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f'  ({len(y_true)} imgs, {elapsed:.1f}s): Acc={acc:.4f} F1={f1:.4f}  {elapsed/len(y_true)*1000:.1f}ms/img')
    print(f'  CM: {cm.tolist()}')
    return acc, f1

for name, path in MODELS:
    try:
        eval_on_mrl(name, path)
    except Exception as e:
        print(f'{name}: FAILED - {e}')
