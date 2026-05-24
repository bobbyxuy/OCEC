#!/usr/bin/env python3
"""Blend real env photos with eye images using heavy blur to avoid boundaries."""
import os, random
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont

SRC_DIR = '/home/bobby/OCEC/glass1_samples'
ENV_DIR = '/home/bobby/OCEC/env_reflection_textures'
OUT_DIR = '/home/bobby/OCEC/env_photo_out'
os.makedirs(OUT_DIR, exist_ok=True)

samples = [
    'awake_0_s0036_05832_1_1_1_0_0_01.png',
    'sleepy_0_s0037_04771_1_1_0_1_0_01.png',
    'awake_2_s0035_00444_0_1_1_0_1_02.png',
    'sleepy_1_s0012_02860_0_1_0_2_1_01.png',
]
VARIANTS = 4

env_files = sorted([f for f in os.listdir(ENV_DIR) if f.endswith('.jpg')])

for fname in samples:
    base = os.path.splitext(fname)[0]
    eye = Image.open(os.path.join(SRC_DIR, fname)).convert('L')
    ew, eh = eye.size

    for v in range(VARIANTS):
        random.seed(42 + v * 1000 + hash(fname) % 10000)

        # Pick random env photo
        env_img = Image.open(os.path.join(ENV_DIR, random.choice(env_files))).convert('RGB')

        # Resize to eye size
        env_img = env_img.resize((ew, eh), Image.LANCZOS)

        # Heavy blur to blend naturally with eye
        blur_r = random.uniform(25, 50)
        env_blur = env_img.filter(ImageFilter.GaussianBlur(radius=blur_r))

        # Convert to grayscale
        env_gray = np.array(env_blur.convert('L'), np.float32)

        # Brightness matching: align mean with eye
        eye_arr = np.array(eye, np.float32)
        eye_mean = eye_arr[eye_arr > 10].mean() if (eye_arr > 10).any() else 128
        env_mean = env_gray[env_gray > 10].mean() if (env_gray > 10).any() else 128
        if env_mean > 0:
            scale = eye_mean / env_mean * random.uniform(0.6, 1.2)
            env_gray = np.clip(env_gray * scale, 0, 255)

        # Add to eye (additive wash)
        strength = random.uniform(0.5, 1.0)
        out = np.clip(eye_arr + env_gray * strength, 0, 255)

        # Small glint spots on top (subtle)
        yy, xx = np.mgrid[:eh, :ew]
        for _ in range(random.randint(3, 7)):
            sx = random.uniform(0.1, 0.9) * ew
            sy = random.uniform(0.1, 0.9) * eh
            sr = random.uniform(3, 15)
            spot = np.exp(-((xx - sx)**2 + (yy - sy)**2) / (2 * sr**2))
            out = np.clip(out + spot * random.uniform(30, 80), 0, 255)

        # Final mild bloom
        out_img = Image.fromarray(out.astype(np.uint8))
        bloom = np.array(out_img.filter(ImageFilter.GaussianBlur(radius=3)), np.float32)
        out = np.clip(out * 0.88 + bloom * 0.12, 0, 255)

        Image.fromarray(out.astype(np.uint8)).save(
            os.path.join(OUT_DIR, f'{base}_env_v{v}.png'))
    print(f'{fname} done')

# Grid
cell, gap, top = 128, 6, 18
cols, rows = 5, len(samples)
canvas = Image.new('L', (cols*cell+(cols-1)*gap, rows*cell+(rows-1)*gap+top), 18)
d = ImageDraw.Draw(canvas)
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
except: font = ImageFont.load_default()
for c, t in enumerate(['orig','env-1','env-2','env-3','env-4']):
    d.text((c*(cell+gap)+4, 2), t, fill=220, font=font)
for r, fname in enumerate(samples):
    base = os.path.splitext(fname)[0]; y = top + r*(cell+gap)
    canvas.paste(Image.open(os.path.join(SRC_DIR, fname)).convert('L').resize((cell,cell),Image.NEAREST),(0,y))
    for v in range(4):
        im = Image.open(os.path.join(OUT_DIR, f'{base}_env_v{v}.png')).convert('L').resize((cell,cell),Image.NEAREST)
        canvas.paste(im, ((v+1)*(cell+gap), y))
GRID = '/home/bobby/OCEC/env_photo_grid.png'
canvas.save(GRID)
print(f'Grid: {GRID}')
