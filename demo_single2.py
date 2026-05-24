#!/usr/bin/env python3
import os, torch, torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image, ImageDraw, ImageFont
import random

backbone = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=64)
feat_dim = backbone.embed_dim
model = nn.Sequential(backbone, nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 2)).cuda()
model.load_state_dict(torch.load('best_dinov2_eye.pth'))
model.eval()
tf = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

samples = []
for cls_name, label in [('awake',0),('sleepy',1)]:
    d = '/home/bobby/eye_datasets/data/test/' + cls_name
    for fname in sorted(os.listdir(d))[:1000]:
        img = Image.open(os.path.join(d, fname)).convert('L').convert('RGB')
        x = tf(img).unsqueeze(0).cuda()
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1)
            conf, pred = p.max(dim=1)
        samples.append((img, int(pred.item()), float(conf.item()), fname, label))

# Pick specific categories
high_open = [s for s in samples if s[3]==0 and s[2]>0.999 and s[4]==0]
high_closed = [s for s in samples if s[3]==1 and s[2]>0.999 and s[4]==1]
uncertain_correct = [s for s in samples if 0.6 < s[2] < 0.8 and s[3]==s[4]]
misclassified = [s for s in samples if s[3] != s[4]]

print("high_open:", len(high_open), "high_closed:", len(high_closed))
print("uncertain_correct:", len(uncertain_correct), "misclassified:", len(misclassified))

picks = random.sample(high_open, 2) + random.sample(high_closed, 2)
if uncertain_correct:
    picks += random.sample(uncertain_correct, min(2, len(uncertain_correct)))
if misclassified:
    picks += misclassified[:2]

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except:
    font = ImageFont.load_default()

for i, (img, pred, conf, fname, true_label) in enumerate(picks):
    big = img.resize((img.width*4, img.height*4), Image.NEAREST)
    draw = ImageDraw.Draw(big)
    pred_str = "OPEN" if pred==0 else "CLOSED"
    true_str = "OPEN" if true_label==0 else "CLOSED"
    correct = pred == true_label
    l1 = "Pred: " + pred_str + " (" + "%.3f" % conf + ")"
    l2 = "True: " + true_str + (" OK" if correct else " WRONG")
    c1 = (0,255,0) if correct else (255,100,100)
    c2 = (0,255,0) if correct else (255,0,255)
    draw.text((5, 5), l1, fill=c1, font=font)
    draw.text((5, 28), l2, fill=c2, font=font)
    outname = "demo_%d_%s_%s_%.3f.png" % (i+1, pred_str, true_str, conf)
    big.save('/home/bobby/OCEC/' + outname)
    print("Saved:", outname, "- conf:", conf, "correct:", correct)
