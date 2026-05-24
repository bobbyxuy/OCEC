#!/usr/bin/env python3
import numpy as np, cv2, os, time, onnxruntime as ort
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

MODEL = 'ocec_l.onnx'
BASE = '/home/bobby/eye_datasets/data'

def predict(sess, inp_name, img_bgr, mode='rgb'):
    if mode == 'gray':
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (40, 24)).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis]
    return float(np.squeeze(sess.run(None, {inp_name: img})[0]))

def eval_split(sess, inp_name, split, mode='rgb'):
    awake_dir = os.path.join(BASE, split, 'awake')
    sleepy_dir = os.path.join(BASE, split, 'sleepy')
    
    y_true, y_pred = [], []
    files = []
    
    for label, d in [(0, awake_dir), (1, sleepy_dir)]:  # 0=open(awake), 1=closed(sleepy)
        fnames = [f for f in os.listdir(d) if f.endswith('.png')]
        for fname in fnames:
            img = cv2.imread(os.path.join(d, fname))
            if img is None: continue
            p = predict(sess, inp_name, img, mode)
            # OCEC: high prob = open (awake=0), low prob = closed (sleepy=1)
            pred = 0 if p >= 0.5 else 1
            y_true.append(label)
            y_pred.append(pred)
            files.append(fname)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    prec = precision_score(y_true, y_pred, average='binary')
    rec = recall_score(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, prec, rec, cm, len(files)

def main():
    print('Loading OCEC...')
    sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
    inp_name = sess.get_inputs()[0].name
    
    for mode in ['rgb', 'gray']:
        print(f'\n===== {mode.upper()} =====')
        for split in ['test', 'val']:
            t0 = time.time()
            acc, f1, prec, rec, cm, n = eval_split(sess, inp_name, split, mode)
            elapsed = time.time() - t0
            print(f'{split:4s} ({n:5d} imgs, {elapsed:.1f}s): Acc={acc:.4f} F1={f1:.4f} P={prec:.4f} R={rec:.4f}  {elapsed/n*1000:.2f}ms/img')
            print(f'  CM: {cm.tolist()}')

if __name__ == '__main__':
    main()
