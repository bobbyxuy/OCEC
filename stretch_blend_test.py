#!/usr/bin/env python3
"""Stretch env image to full eye size, alpha blend - test for boundary."""
import os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC_DIR = '/home/bobby/OCEC/glass1_samples'
REFL_DIR = '/home/bobby/OCEC/blender_out3/reflections'
OUT_DIR = '/home/bobby/OCEC/stretch_test'
os.makedirs(OUT_DIR, exist_ok=True)

samples = [
    'awake_0_s0036_05832_1_1_1_0_0_01.png',
    'sleepy_0_s0037_04771_1_1_0_1_0_01.png',
    'awake_2_s0035_00444_0_1_1_0_1_02.png',
    'sleepy_1_s0012_02860_0_1_0_2_1_01.png',
]
VARIANTS = 4

for fname in samples:
    base = os.path.splitext(fname)[0]
    eye = Image.open(os.path.join(SRC_DIR, fname)).convert('L')
    ew, eh = eye.size
    
    for v in range(VARIANTS):
        random.seed(42 + v * 1000 + hash(fname) % 10000)
        refl = Image.open(os.path.join(REFL_DIR, f'{base}_refl_v{v}.png')).convert('L')
        # Scale up 1.5x so reflection fills more of the frame, then crop center
        rw, rh = refl.size
        refl = refl.resize((int(rw * 1.5), int(rh * 1.5)), Image.LANCZOS)
        # Crop center to original size
        cw, ch = refl.size
        ox, oy = (cw - ew) // 2, (ch - eh) // 2
        refl = refl.crop((ox, oy, ox + ew, oy + eh))
        
        eye_arr = np.array(eye, np.float32)
        refl_arr = np.array(refl, np.float32)
        
        # Brighten reflection
        refl_arr = refl_arr * random.uniform(1.5, 3.0)
        
        # Uniform alpha blend
        alpha = random.uniform(0.4, 0.7)
        out = eye_arr * (1 - alpha) + refl_arr * alpha
        
        # Clip and save
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(
            os.path.join(OUT_DIR, f'{base}_stretch_v{v}.png'))
    print(f'{fname} done')

# Grid
cell, gap, top = 128, 6, 18
cols, rows = 5, len(samples)
canvas = Image.new('L', (cols*cell+(cols-1)*gap, rows*cell+(rows-1)*gap+top), 18)
d = ImageDraw.Draw(canvas)
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
except: font = ImageFont.load_default()
for c, t in enumerate(['orig','str-1','str-2','str-3','str-4']):
    d.text((c*(cell+gap)+4, 2), t, fill=220, font=font)
for r, fname in enumerate(samples):
    base = os.path.splitext(fname)[0]; y = top + r*(cell+gap)
    canvas.paste(Image.open(os.path.join(SRC_DIR, fname)).convert('L').resize((cell,cell),Image.NEAREST),(0,y))
    for v in range(4):
        im = Image.open(os.path.join(OUT_DIR, f'{base}_stretch_v{v}.png')).convert('L').resize((cell,cell),Image.NEAREST)
        canvas.paste(im,((v+1)*(cell+gap),y))
GRID = '/home/bobby/OCEC/stretch_grid.png'
canvas.save(GRID)
print(f'Grid: {GRID}')
