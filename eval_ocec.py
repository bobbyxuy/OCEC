#!/usr/bin/env python3
"""Evaluate OCEC on the closed-open-eyes dataset (HuggingFace).
Tests both RGB and grayscale (simulated NIR) inputs.
"""
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import time

MODEL_PATH = 'ocec_l.onnx'

def predict(sess, inp_name, pil_img, mode='rgb'):
    """Predict P(open) for a PIL image.
    mode: 'rgb' or 'gray'
    """
    if mode == 'gray':
        img = pil_img.convert('L')
        img = img.convert('RGB')  # 1ch -> 3ch
    else:
        img = pil_img.convert('RGB')
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
    img_array = np.expand_dims(img_array, 0)
    
    result = sess.run(None, {inp_name: img_array})[0]
    return float(np.squeeze(result))

def main():
    print('Loading OCEC model...')
    sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    inp_name = sess.get_inputs()[0].name
    print(f'  Input: {sess.get_inputs()[0].shape}')
    
    print('Loading dataset...')
    ds = load_dataset('MichalMlodawski/closed-open-eyes', split='train')
    print(f'  Total: {len(ds)} samples')
    
    # Check label mapping
    features = ds.features
    print(f'  Features: {features}')
    
    # Determine label mapping (0=open, 1=closed or vice versa)
    label_names = features['label'].names if hasattr(features['label'], 'names') else None
    print(f'  Label names: {label_names}')
    
    results = {}
    for mode in ['rgb', 'gray']:
        print(f'\n=== Mode: {mode.upper()} ===')
        labels_true = []
        labels_pred = []
        probs = []
        
        start = time.time()
        for i, sample in enumerate(ds):
            img = sample['image']
            label = sample['label']  # 0 or 1
            p_open = predict(sess, inp_name, img, mode=mode)
            
            # OCEC output: P(open), threshold 0.5
            pred_open = 1 if p_open >= 0.5 else 0  # 1=open, 0=closed
            
            labels_true.append(label)
            probs.append(p_open)
            
            # Map pred to dataset label space
            # If label_names = ['Closed', 'Open'], then 0=closed, 1=open -> pred_open maps directly
            # If label_names = ['Open', 'Closed'], then 0=open, 1=closed -> pred_open needs flip
            if label_names and label_names[0].lower() == 'closed':
                labels_pred.append(pred_open)
            elif label_names and label_names[0].lower() == 'open':
                labels_pred.append(1 - pred_open)
            else:
                # Assume 0=open, 1=closed based on common convention
                labels_pred.append(1 - pred_open)  # pred_open=1 -> dataset label 0 (open)
            
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                print(f'  [{i+1}/{len(ds)}] elapsed={elapsed:.1f}s, avg={elapsed/(i+1)*1000:.1f}ms/img')
        
        elapsed = time.time() - start
        
        # Calculate metrics
        acc = accuracy_score(labels_true, labels_pred)
        f1 = f1_score(labels_true, labels_pred, average='binary')
        prec = precision_score(labels_true, labels_pred, average='binary')
        rec = recall_score(labels_true, labels_pred, average='binary')
        cm = confusion_matrix(labels_true, labels_pred)
        
        print(f'\nResults ({mode.upper()}):')
        print(f'  Accuracy:  {acc:.4f}')
        print(f'  F1 Score:  {f1:.4f}')
        print(f'  Precision: {prec:.4f}')
        print(f'  Recall:    {rec:.4f}')
        print(f'  Speed:     {elapsed/len(ds)*1000:.2f} ms/img')
        print(f'  Confusion Matrix:')
        print(f'    {cm}')
        
        results[mode] = {
            'accuracy': acc,
            'f1': f1,
            'precision': prec,
            'recall': rec,
            'speed_ms': elapsed/len(ds)*1000,
            'cm': cm,
            'label_names': label_names,
        }
    
    # Summary
    print(f'\n========== SUMMARY ==========')
    for mode, r in results.items():
        print(f'{mode.upper():5s}: Acc={r["accuracy"]:.4f}  F1={r["f1"]:.4f}  P={r["precision"]:.4f}  R={r["recall"]:.4f}  {r["speed_ms"]:.2f}ms/img')

if __name__ == '__main__':
    main()
