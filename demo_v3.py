#!/usr/bin/env python3
import os, torch, torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import timm

backbone = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=64)
feat_dim = backbone.embed_dim
model = nn.Sequential(backbone, nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 2))
model.load_state_dict(torch.load('best_dinov2_eye.pth', map_location='cpu'))
model.eval()
tf = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)

samples = [
    ("awake", 0, "s0001_01846_0_0_1_0_0_01.png"),
    ("awake", 0, "s0001_02017_0_0_1_0_0_01.png"),
    ("awake", 0, "s0001_02103_0_0_1_0_0_01.png"),
    ("sleepy", 1, "s0001_00001_0_0_0_0_0_01.png"),
    ("sleepy", 1, "s0001_00752_0_0_0_0_0_01.png"),
    ("sleepy", 1, "s0009_00183_0_0_0_0_0_01.png"),
]

for idx, (cls, true_label, fname) in enumerate(samples):
    path = "/home/bobby/eye_datasets/data/test/" + cls + "/" + fname
    img = Image.open(path).convert("L").convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)
        conf = float(p.max().item())
        pred = int(p.argmax().item())

    big = img.resize((512, 512), Image.NEAREST)
    draw = ImageDraw.Draw(big)

    pred_str = "OPEN" if pred == 0 else "SHUT"
    true_str = "OPEN" if true_label == 0 else "SHUT"
    correct = (pred == true_label) and conf >= 0.85

    if conf < 0.85:
        l1 = "P:UNSURE " + "%.4f" % conf
        c1 = (200, 200, 0)
    else:
        l1 = "P:" + pred_str + " " + "%.4f" % conf
        c1 = (0, 255, 0) if pred == 0 else (255, 80, 80)

    if correct:
        l2 = "T:" + true_str + " OK"
        c2 = (0, 255, 0)
    else:
        l2 = "T:" + true_str + " WRONG"
        c2 = (255, 0, 255)

    draw.rectangle([0, 0, 512, 60], fill=(0, 0, 0))
    draw.text((10, 5), l1, fill=c1, font=font)
    draw.text((10, 33), l2, fill=c2, font=font_sm)

    outname = "demo_v3_%d.png" % (idx + 1)
    big.save("/home/bobby/OCEC/" + outname)
    print(outname)
