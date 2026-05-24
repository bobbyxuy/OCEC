#!/usr/bin/env python3
"""Blurry env overlay + bright glint spots - 隐约可见环境内容."""
import os, random, math
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

        env_img = Image.open(os.path.join(ENV_DIR, random.choice(env_files))).convert('RGB')
        env_img = env_img.resize((ew, eh), Image.LANCZOS)

        # Very heavy blur - env content becomes vague shapes
        blur_r = random.uniform(20, 40)
        env_blur = env_img.filter(ImageFilter.GaussianBlur(radius=blur_r))
        env_gray = np.array(env_blur.convert('L'), np.float32)

        eye_arr = np.array(eye, np.float32)

        # Normalize and blend
        eye_norm = (eye_arr - eye_arr.min()) / max(1, eye_arr.max() - eye_arr.min())
        env_norm = (env_gray - env_gray.min()) / max(1, env_gray.max() - env_gray.min())
        blend = random.uniform(0.40, 0.60)
        combined = eye_norm * (1 - blend) + env_norm * blend
        out = combined * 255.0

        # Ellipse glint spots
        yy, xx = np.mgrid[:eh, :ew]
        for _ in range(random.randint(3, 6)):
            sx = random.uniform(0.1, 0.9) * ew
            sy = random.uniform(0.1, 0.9) * eh
            sr_x = random.uniform(30, 80)
            sr_y = random.uniform(10, 35)
            angle = random.uniform(-0.3, 0.3)
            dx = xx - sx; dy = yy - sy
            rx = dx * math.cos(angle) + dy * math.sin(angle)
            ry = -dx * math.sin(angle) + dy * math.cos(angle)
            spot = np.exp(-(rx**2 / (2 * sr_x**2) + ry**2 / (2 * sr_y**2)))
            out = np.clip(out + spot * random.uniform(100, 180), 0, 255)
        
        # Rectangular glint patches with soft edges
        for _ in range(random.randint(8, 12)):
            rw = random.uniform(80, 220)
            rh = random.uniform(40, 120)
            rx0 = random.uniform(0.0, 0.65) * ew
            ry0 = random.uniform(0.0, 0.65) * eh
            edge = random.uniform(1, 4)
            rect = np.ones((eh, ew), np.float32)
            rect = rect * np.exp(-((xx - rx0)**2) / (2 * edge**2))
            rect = rect * np.exp(-((xx - rx0 - rw)**2) / (2 * edge**2))
            rect = rect * np.exp(-((yy - ry0)**2) / (2 * edge**2))
            rect = rect * np.exp(-((yy - ry0 - rh)**2) / (2 * edge**2))
            out = np.clip(out + rect * random.uniform(110, 200), 0, 255)

        # Bloom
        out_img = Image.fromarray(out.astype(np.uint8))
        bloom = np.array(out_img.filter(ImageFilter.GaussianBlur(radius=4)), np.float32)
        out = np.clip(out * 0.85 + bloom * 0.15, 0, 255)

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
