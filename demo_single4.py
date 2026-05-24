#!/usr/bin/env python3
import os, torch, torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import timm, random
random.seed(42)

backbone = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=64)
feat_dim = backbone.embed_dim
model = nn.Sequential(backbone, nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 2))
model.load_state_dict(torch.load('best_dinov2_eye.pth', map_location='cpu'))
model.eval()
tf = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

high_open = []
high_closed = []
uncertain_correct = []
misclassified = []

for cls_name, label in [('awake',0),('sleepy',1)]:
    d = '/home/bobby/eye_datasets/data/test/' + cls_name
    fnames = sorted(os.listdir(d))[:500]
    for fname in fnames:
        img = Image.open(os.path.join(d, fname)).convert('L').convert('RGB')
        x = tf(img).unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1)
            conf = float(p.max().item())
            pred = int(p.argmax().item())
        
        entry = (img, pred, conf, fname, label)
        correct = (pred == label)
        
        if correct and conf > 0.999:
            if label == 0:
                high_open.append(entry)
            else:
                high_closed.append(entry)
        elif correct and 0.6 < conf < 0.8:
            uncertain_correct.append(entry)
        elif not correct:
            misclassified.append(entry)

print("Categories: high_open=%d high_closed=%d uncertain_correct=%d misclassified=%d" % (
    len(high_open), len(high_closed), len(uncertain_correct), len(misclassified)))

picks = random.sample(high_open, 2) + random.sample(high_closed, 2)
picks += random.sample(uncertain_correct, min(2, len(uncertain_correct)))
picks += misclassified[:2]

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
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
    draw.text((5, 30), l2, fill=c2, font=font)
    outname = "demo_%d_%s_%s_%.3f.png" % (i+1, pred_str, true_str, conf)
    big.save('/home/bobby/OCEC/' + outname)
    print("Saved:", outname)
