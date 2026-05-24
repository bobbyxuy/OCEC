#!/usr/bin/env python3
import os, torch, torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
from collections import Counter

backbone = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=64)
feat_dim = backbone.embed_dim
model = nn.Sequential(backbone, nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 2)).cuda()
model.load_state_dict(torch.load('best_dinov2_eye.pth'))
model.eval()
tf = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

wrong = []
for cls_name, label in [('awake',0),('sleepy',1)]:
    d = '/home/bobby/eye_datasets/data/test/' + cls_name
    for fname in sorted(os.listdir(d)):
        img = Image.open(os.path.join(d, fname)).convert('L').convert('RGB')
        x = tf(img).unsqueeze(0).cuda()
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1)
            conf = float(p.max().item())
            pred = int(p.argmax().item())
        if pred != label:
            parts = fname.replace('.png','').split('_')
            wrong.append({
                'fname': fname,
                'true': 'awake' if label==0 else 'sleepy',
                'pred': 'awake' if pred==0 else 'sleepy',
                'conf': conf,
                'subject': parts[0],
                'light': parts[2] if len(parts)>2 else '?',
                'glass': parts[3] if len(parts)>3 else '?',
                'motion': parts[4] if len(parts)>4 else '?',
                'eye': parts[5] if len(parts)>5 else '?',
                'w': img.width, 'h': img.height,
            })

total = 16981
print('Total misclassified: %d / %d (%.2f%%)' % (len(wrong), total, len(wrong)/total*100))

print('\n=== By true label ===')
for k,v in Counter(w['true'] for w in wrong).items():
    t = 8591 if k=='awake' else 8390
    print('  %s: %d / %d (%.2f%%)' % (k, v, t, v/t*100))

print('\n=== By subject (top 10) ===')
for subj, cnt in Counter(w['subject'] for w in wrong).most_common(10):
    print('  %s: %d' % (subj, cnt))

print('\n=== By light ===')
for k,v in Counter(w['light'] for w in wrong).items():
    print('  light=%s: %d' % (k, v))

print('\n=== By glass ===')
for k,v in Counter(w['glass'] for w in wrong).items():
    print('  glass=%s: %d' % (k, v))

print('\n=== By motion ===')
for k,v in Counter(w['motion'] for w in wrong).items():
    print('  motion=%s: %d' % (k, v))

print('\n=== By eye ===')
for k,v in Counter(w['eye'] for w in wrong).items():
    print('  eye=%s: %d' % (k, v))

print('\n=== By image size ===')
for k,v in Counter((w['w'],w['h']) for w in wrong).most_common(10):
    print('  %dx%d: %d' % (k[0], k[1], v))

print('\n=== By confidence ===')
low = sum(1 for w in wrong if w['conf'] < 0.6)
mid = sum(1 for w in wrong if 0.6 <= w['conf'] < 0.8)
high = sum(1 for w in wrong if w['conf'] >= 0.8)
print('  conf<0.6: %d, 0.6-0.8: %d, >=0.8: %d' % (low, mid, high))

print('\n=== All samples (sorted by conf desc) ===')
for w in sorted(wrong, key=lambda x: -x['conf']):
    print('  %s true=%s pred=%s conf=%.3f %dx%d sub=%s L=%s G=%s M=%s E=%s' % (
        w['fname'], w['true'], w['pred'], w['conf'], w['w'], w['h'],
        w['subject'], w['light'], w['glass'], w['motion'], w['eye']))
