#!/usr/bin/env python3
import os, random
import numpy as np
from PIL import Image, ImageFilter

SRC_DIR = '/home/bobby/OCEC/glass1_samples'
REFL_DIR = '/home/bobby/OCEC/blender_out3/reflections'
OUT_DIR = '/home/bobby/OCEC/blender_out4'
os.makedirs(OUT_DIR, exist_ok=True)

samples = [
    'awake_0_s0036_05832_1_1_1_0_0_01.png',
    'sleepy_0_s0037_04771_1_1_0_1_0_01.png',
    'awake_2_s0035_00444_0_1_1_0_1_02.png',
    'sleepy_1_s0012_02860_0_1_0_2_1_01.png',
]
VARIANTS = 4

def make_random_mask(h, w):
    """Full coverage mask - no boundary, just intensity variation."""
    x_norm = np.linspace(0, 1, w)
    y_norm = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x_norm, y_norm)
    variation = 0.7 + 0.3 * np.sin(xx * np.pi * random.uniform(1, 3) + random.uniform(0, 6.28))
    variation *= 0.8 + 0.2 * np.sin(yy * np.pi * random.uniform(1, 2) + random.uniform(0, 6.28))
    mask = np.clip(variation, 0, 1)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(20, 40)))
    return np.array(mask_img, np.float32) / 255.0

import math

for fname in samples:
    base = os.path.splitext(fname)[0]
    eye = np.array(Image.open(os.path.join(SRC_DIR, fname)).convert('L'), np.float32)
    h, w = eye.shape
    yy, xx = np.mgrid[:h, :w]
    
    for v in range(VARIANTS):
        random.seed(42 + v * 1000 + hash(fname) % 10000)
        refl_raw = np.array(Image.open(os.path.join(REFL_DIR, f'{base}_refl_v{v}.png')).convert('L').transpose(Image.FLIP_TOP_BOTTOM), np.float32)
        # Fill black areas with eye image so no boundary exists
        black_mask = refl_raw < 15
        refl_raw[black_mask] = eye[black_mask]
        
        refl = refl_raw
        mask = make_random_mask(h, w)
        
        # ADDITIVE blend: reflection adds light on top of eye, no replacement
        # Brighten reflection to wash out eye detail
        refl_boost = refl * random.uniform(1.5, 3.0)
        add_strength = mask * random.uniform(0.6, 1.2)
        out = np.clip(eye + refl_boost * add_strength, 0, 255)
        
        # Heavy specular glint spots (6-12)
        for _ in range(random.randint(6, 12)):
            sx = random.uniform(0.1, 0.9) * w
            sy = random.uniform(0.1, 0.9) * h
            sr = random.uniform(3, 20)
            strength = random.uniform(0.4, 1.0)
            # Elongated horizontally (streak-like)
            sx_r = random.uniform(0.5, 2.0)
            sy_r = random.uniform(0.5, 1.0)
            spot = np.exp(-(((xx-sx)/sx_r)**2 + ((yy-sy)/sy_r)**2) / (2 * sr**2))
            spot_mask = spot * mask
            # Clip to create hard saturated highlights
            clipped = np.clip(spot_mask * strength * 255, 0, 255)
            clipped[clipped > random.randint(160, 200)] = 255
            out = np.clip(out + clipped, 0, 255)
        
        # Strong bloom on highlights
        out_img = Image.fromarray(out.astype(np.uint8))
        bloom1 = np.array(out_img.filter(ImageFilter.GaussianBlur(radius=3)), np.float32)
        bloom2 = np.array(out_img.filter(ImageFilter.GaussianBlur(radius=8)), np.float32)
        out = np.clip(out * 0.75 + bloom1 * 0.15 + bloom2 * 0.10, 0, 255)
        
        out_path = os.path.join(OUT_DIR, f'{base}_comp_v{v}.png')
        Image.fromarray(out.astype(np.uint8)).save(out_path)
        print(f'{fname} v{v}')

# Grid
from PIL import ImageDraw, ImageFont
cell, gap, top = 128, 6, 18
cols, rows = 5, len(samples)
canvas = Image.new('L', (cols*cell+(cols-1)*gap, rows*cell+(rows-1)*gap+top), 18)
d = ImageDraw.Draw(canvas)
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
except: font = ImageFont.load_default()
for c, t in enumerate(['orig','v4-1','v4-2','v4-3','v4-4']):
    d.text((c*(cell+gap)+4, 2), t, fill=220, font=font)
for r, fname in enumerate(samples):
    base = os.path.splitext(fname)[0]; y = top + r*(cell+gap)
    canvas.paste(Image.open(os.path.join(SRC_DIR, fname)).convert('L').resize((cell, cell), Image.NEAREST), (0, y))
    for v in range(4):
        im = Image.open(os.path.join(OUT_DIR, f'{base}_comp_v{v}.png')).convert('L').resize((cell, cell), Image.NEAREST)
        canvas.paste(im, ((v+1)*(cell+gap), y))
GRID = '/home/bobby/OCEC/comp_v4_grid.png'
canvas.save(GRID)
print(f'Grid: {GRID}')
