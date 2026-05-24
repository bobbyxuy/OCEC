#!/usr/bin/env python3
import os, numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch, torch.nn as nn
from torchvision import transforms
import timm

backbone = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=64)
feat_dim = backbone.embed_dim
model = nn.Sequential(backbone, nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 2)).cuda()
model.load_state_dict(torch.load('best_dinov2_eye.pth'))
model.eval()

tf = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

samples = []
for cls_name, label in [('awake',0),('sleepy',1)]:
    d = '/home/bobby/eye_datasets/data/test/' + cls_name
    fnames = sorted(os.listdir(d))[:500]
    for fname in fnames:
        img = Image.open(os.path.join(d, fname)).convert('L').convert('RGB')
        x = tf(img).unsqueeze(0).cuda()
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1)
            conf, pred = p.max(dim=1)
        samples.append((img, int(pred.item()), float(conf.item()), fname, label, cls_name))

samples.sort(key=lambda s: s[2])

uncertain = [s for s in samples if 0.5 < s[2] < 0.85][:3]
misclassified = [s for s in samples if s[3] != s[4]][:3]
high_open = [s for s in samples if s[3]==0 and s[2]>0.99 and s[4]==0][-3:]
high_closed = [s for s in samples if s[3]==1 and s[2]>0.99 and s[4]==1][-3:]
selected = high_open + high_closed + uncertain + misclassified

cols = len(selected)
cell = 120
pad = 10
grid_w = cols * (cell + pad) + pad
grid_h = cell + 60
grid = Image.new('RGB', (grid_w, grid_h), (40, 40, 40))
draw = ImageDraw.Draw(grid)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
except:
    font = font_sm = ImageFont.load_default()

for i, (img, pred, conf, fname, true_label, cls_name) in enumerate(selected):
    x_off = pad + i * (cell + pad)
    img_resized = img.resize((cell, cell))
    grid.paste(img_resized, (x_off, 30))

    if conf < 0.85:
        label_text = "UNCERTAIN %.2f" % conf
        color = (200, 200, 0)
    else:
        lstr = "OPEN" if pred == 0 else "CLOSED"
        label_text = "%s %.2f" % (lstr, conf)
        color = (0, 200, 0) if pred == 0 else (200, 0, 0)

    if pred != true_label:
        draw.rectangle([x_off-2, 28, x_off+cell+1, 31+cell], outline=(255, 0, 255), width=2)

    draw.text((x_off, 2), label_text, fill=color, font=font)
    draw.text((x_off, cell + 35), fname[:20], fill=(150,150,150), font=font_sm)
    true_text = "true:%s" % ("open" if true_label == 0 else "closed")
    tc = (100,200,100) if pred == true_label else (255,100,100)
    draw.text((x_off, cell + 48), true_text, fill=tc, font=font_sm)

grid.save('/home/bobby/OCEC/demo_grid.png')
print("Saved demo_grid.png: %dx%d" % (grid_w, grid_h))
print("3 high-conf open | 3 high-conf closed | 3 uncertain | 3 misclassified")
