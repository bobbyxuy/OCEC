#!/usr/bin/env python3
import os, torch, torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

backbone = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=64)
feat_dim = backbone.embed_dim
model = nn.Sequential(backbone, nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 2))
sd = torch.load('best_dinov2_eye.pth', map_location='cpu')
model.load_state_dict(sd)
model.eval()
tf = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

for cls_name, label in [('awake',0),('sleepy',1)]:
    d = '/home/bobby/eye_datasets/data/test/' + cls_name
    fname = sorted(os.listdir(d))[0]
    img = Image.open(os.path.join(d, fname)).convert('L').convert('RGB')
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        p = torch.softmax(logits, dim=1)
    print(cls_name, fname)
    print("  logits:", logits[0].tolist())
    print("  prob:", p[0].tolist())
    print("  pred:", p.argmax().item(), "true:", label)
