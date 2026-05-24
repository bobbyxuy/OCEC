#!/usr/bin/env python3
import os, random
import numpy as np
from PIL import Image, ImageFilter

SRC_DIR = '/home/bobby/OCEC/glass1_samples'
REFL_DIR = '/home/bobby/OCEC/blender_out3/reflections'
OUT_DIR = '/home/bobby/OCEC/blender_out3'

samples = [
    'awake_0_s0036_05832_1_1_1_0_0_01.png',
    'sleepy_0_s0037_04771_1_1_0_1_0_01.png',
    'awake_2_s0035_00444_0_1_1_0_1_02.png',
    'sleepy_1_s0012_02860_0_1_0_2_1_01.png',
]
VARIANTS = 4

for fname in samples:
    base = os.path.splitext(fname)[0]
    eye = np.array(Image.open(os.path.join(SRC_DIR, fname)).convert('L'), np.float32)
    h, w = eye.shape
    
    for v in range(VARIANTS):
        random.seed(42 + v * 1000 + hash(fname) % 10000)
        refl = np.array(Image.open(os.path.join(REFL_DIR, f'{base}_refl_v{v}.png')).convert('L'), np.float32)
        
        # Elliptical lens mask, biased lower
        cy = int(h * random.uniform(0.50, 0.60))
        cx = int(w * random.uniform(0.45, 0.55))
        rx = int(w * random.uniform(0.35, 0.50))
        ry = int(h * random.uniform(0.40, 0.58))
        yy, xx = np.ogrid[:h, :w]
        dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        
        edge_w = random.uniform(0.05, 0.15)
        mask = np.clip(1.0 - (dist - (1.0 - edge_w)) / edge_w, 0, 1)
        mask[dist > 1.0] = 0
        
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=random.uniform(3, 8)))
        mask = np.array(mask_img, np.float32) / 255.0
        
        # Gradient: upper stronger
        y_grad = np.linspace(1.0, 0.55, h)[:, None]
        mask = mask * y_grad
        
        # Blend reflection onto eye
        alpha = mask * random.uniform(0.25, 0.50)
        out = eye * (1.0 - alpha) + refl * alpha
        
        # Specular glint spots
        for _ in range(random.randint(2, 5)):
            sx = random.randint(int(w*0.15), int(w*0.85))
            sy = random.randint(int(h*0.1), int(h*0.55))
            sr = random.randint(4, 16)
            strength = random.uniform(0.3, 0.7)
            spot = np.exp(-((xx-sx)**2 + (yy-sy)**2) / (2*sr**2))
            out = np.clip(out + spot * strength * mask * 255, 0, 255)
        
        # Bloom
        out_img = Image.fromarray(out.astype(np.uint8))
        bloom = np.array(out_img.filter(ImageFilter.GaussianBlur(radius=4)), np.float32)
        out = np.clip(out * 0.85 + bloom * 0.15, 0, 255)
        
        out_path = os.path.join(OUT_DIR, f'{base}_composite_v{v}.png')
        Image.fromarray(out.astype(np.uint8)).save(out_path)
        print(f'Composited {fname} v{v}')

print(f'Done! {len([f for f in os.listdir(OUT_DIR) if "composite" in f])} composites')
