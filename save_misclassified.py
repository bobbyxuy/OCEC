#!/usr/bin/env python3
"""Extract misclassified samples, draw labels, save to organized directories."""
import os, shutil, torch, torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import timm

backbone = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=64)
feat_dim = backbone.embed_dim
model = nn.Sequential(backbone, nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 2)).cuda()
model.load_state_dict(torch.load('best_dinov2_eye.pth'))
model.eval()
tf = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except:
    font = font_sm = ImageFont.load_default()

out_root = '/home/bobby/OCEC/misclassified'
if os.path.exists(out_root):
    shutil.rmtree(out_root)

count = 0
for cls_name, label in [('awake',0),('sleepy',1)]:
    d = '/home/bobby/eye_datasets/data/test/' + cls_name
    for fname in sorted(os.listdir(d)):
        img = Image.open(os.path.join(d, fname)).convert('L').convert('RGB')
        x = tf(img).unsqueeze(0).cuda()
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1)
            conf = float(p.max().item())
            pred = int(p.argmax().item())

        if pred == label:
            continue

        # Parse metadata
        parts = fname.replace('.png','').split('_')
        glass = parts[3] if len(parts) > 3 else '0'
        light = parts[2] if len(parts) > 2 else '0'
        motion = parts[4] if len(parts) > 4 else '0'
        eye = parts[5] if len(parts) > 5 else '0'

        # Subdirectory: trueLabel/prediction
        true_str = 'awake' if label == 0 else 'sleepy'
        pred_str = 'awake' if pred == 0 else 'sleepy'
        sub_dir = os.path.join(out_root, '%s_pred_%s' % (true_str, pred_str))
        os.makedirs(sub_dir, exist_ok=True)

        # Draw label on 512x512 image
        big = img.resize((512, 512), Image.NEAREST)
        draw = ImageDraw.Draw(big)

        l1 = "P:%s %.4f" % (pred_str, conf)
        c1 = (0, 255, 0) if pred == 0 else (255, 80, 80)
        l2 = "T:%s WRONG" % true_str
        c2 = (255, 0, 255)
        l3 = "G=%s L=%s M=%s E=%s" % (glass, light, motion, eye)
        c3 = (200, 200, 200)

        draw.rectangle([0, 0, 512, 72], fill=(0, 0, 0))
        draw.text((10, 5), l1, fill=c1, font=font)
        draw.text((10, 30), l2, fill=c2, font=font_sm)
        draw.text((10, 52), l3, fill=c3, font=font_sm)

        out_path = os.path.join(sub_dir, fname)
        big.save(out_path)
        count += 1

print("Saved %d images to %s" % (count, out_root))
for sub in sorted(os.listdir(out_root)):
    n = len(os.listdir(os.path.join(out_root, sub)))
    print("  %s/: %d images" % (sub, n))
