#!/usr/bin/env python3
"""Generate a clear demo grid showing eye classification results."""
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

# Pick 2 per category
uncertain = [s for s in samples if 0.5 < s[2] < 0.85][:2]
misclassified = [s for s in samples if s[3] != s[4]][:2]
high_open = [s for s in samples if s[3]==0 and s[2]>0.99 and s[4]==0][-2:]
high_closed = [s for s in samples if s[3]==1 and s[2]>0.99 and s[4]==1][-2:]

groups = [
    ("HIGH CONF OPEN (Pred=OPEN, True=OPEN)", high_open, (0,200,0)),
    ("HIGH CONF CLOSED (Pred=CLOSED, True=CLOSED)", high_closed, (200,0,0)),
    ("UNCERTAIN (Conf<0.85)", uncertain, (200,200,0)),
    ("MISCLASSIFIED (Pred!=True)", misclassified, (255,0,255)),
]

cell = 200
pad = 20
label_h = 50
true_h = 40
row_h = cell + label_h + true_h
grid_w = 2 * cell + 3 * pad
grid_h = len(groups) * (row_h + pad) + pad

grid = Image.new('RGB', (grid_w, grid_h), (30, 30, 30))
draw = ImageDraw.Draw(grid)

try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except:
    font_title = font_label = font_sm = ImageFont.load_default()

y_cursor = pad

for group_title, items, title_color in groups:
    # Group title
    draw.text((pad, y_cursor), group_title, fill=title_color, font=font_title)
    y_cursor += 25

    for i, (img, pred, conf, fname, true_label, cls_name) in enumerate(items):
        x_off = pad + i * (cell + pad)

        # Draw image
        img_resized = img.resize((cell, cell), Image.NEAREST)
        # Add border
        border_color = (255, 0, 255) if pred != true_label else (100, 100, 100)
        border_w = 3 if pred != true_label else 1
        draw.rectangle([x_off-border_w, y_cursor-border_w, x_off+cell+border_w, y_cursor+cell+border_w], outline=border_color, width=border_w)
        grid.paste(img_resized, (x_off, y_cursor))

        # Prediction label (below image)
        pred_str = "OPEN" if pred == 0 else "CLOSED"
        if conf < 0.85:
            pred_text = "Pred: UNCERTAIN (%.3f)" % conf
            pred_color = (200, 200, 0)
        else:
            pred_text = "Pred: %s (%.3f)" % (pred_str, conf)
            pred_color = (0, 200, 0) if pred == 0 else (200, 0, 0)
        draw.text((x_off, y_cursor + cell + 5), pred_text, fill=pred_color, font=font_label)

        # True label
        true_str = "OPEN" if true_label == 0 else "CLOSED"
        true_color = (100, 200, 100)
        if pred != true_label:
            true_color = (255, 100, 100)
            draw.text((x_off, y_cursor + cell + 25), "TRUE: %s [WRONG!]" % true_str, fill=true_color, font=font_sm)
        else:
            draw.text((x_off, y_cursor + cell + 25), "True: %s" % true_str, fill=true_color, font=font_sm)

    y_cursor += row_h + pad

grid.save('/home/bobby/OCEC/demo_grid_v2.png')
print("Saved demo_grid_v2.png: %dx%d" % (grid_w, grid_h))
