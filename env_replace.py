#!/usr/bin/env python3
"""Replace eye brightness with env brightness - preserves texture, shows env structure."""
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

        env_img = Image.open(os.path.join(ENV_DIR, random.choice(env_files))).convert('RGB')
        env_img = env_img.resize((ew, eh), Image.LANCZOS)

        # Light blur so env edges soften but content is clear
        blur_r = random.uniform(6, 14)
        env_blur = env_img.filter(ImageFilter.GaussianBlur(radius=blur_r))
        env_gray = np.array(env_blur.convert('L'), np.float32)

        eye_arr = np.array(eye, np.float32)

        # Normalize each to 0-1 range
        eye_norm = (eye_arr - eye_arr.min()) / max(1, eye_arr.max() - eye_arr.min())
        env_norm = (env_gray - env_gray.min()) / max(1, env_gray.max() - env_gray.min())

        # Blend: replace eye brightness partially with env brightness
        # This preserves eye structure (from eye_norm) but uses env contrast
        blend = random.uniform(0.4, 0.7)
        combined = eye_norm * (1 - blend) + env_norm * blend

        # Scale back to 0-255
        out = combined * 255.0
        out = np.clip(out, 0, 255)

        # Mild bloom
        out_img = Image.fromarray(out.astype(np.uint8))
        bloom = np.array(out_img.filter(ImageFilter.GaussianBlur(radius=2)), np.float32)
        out = np.clip(out * 0.92 + bloom * 0.08, 0, 255)

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
