#!/usr/bin/env python3
"""Pure procedural NIR glass reflection - single image, zero boundary."""
import os, random, math
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont

SRC_DIR = '/home/bobby/OCEC/glass1_samples'
OUT_DIR = '/home/bobby/OCEC/procedural_out'
os.makedirs(OUT_DIR, exist_ok=True)

samples = [
    'awake_0_s0036_05832_1_1_1_0_0_01.png',
    'sleepy_0_s0037_04771_1_1_0_1_0_01.png',
    'awake_2_s0035_00444_0_1_1_0_1_02.png',
    'sleepy_1_s0012_02860_0_1_0_2_1_01.png',
]
VARIANTS = 4

def perlin_like(h, w, scale, seed):
    """Generate smooth random field at given scale."""
    random.seed(seed)
    # Low-res random grid, then upscale with smooth interpolation
    sh, sw = max(2, h // scale), max(2, w // scale)
    grid = np.random.rand(sh, sw).astype(np.float32)
    img = Image.fromarray((grid * 255).astype(np.uint8))
    img = img.resize((w, h), Image.BICUBIC)
    arr = np.array(img, np.float32) / 255.0
    return arr

def synthesize(eye_arr):
    h, w = eye_arr.shape
    yy, xx = np.mgrid[:h, :w]
    
    # 1. Large-scale environment reflection (smooth bright variations)
    # Multiple octaves of smooth noise blended together
    env = np.zeros((h, w), np.float32)
    for scale in [8, 16, 32, 64]:
        env += perlin_like(h, w, scale, random.randint(0, 99999)) * (scale / 64.0)
    env = env / env.max()  # normalize to 0-1
    
    # Bias toward lower half
    y_bias = np.clip((yy / h - 0.3) / 0.5, 0, 1)
    env = env * (0.5 + 0.5 * y_bias)
    
    # 2. Add environment reflection as bright wash on top of eye
    env_strength = random.uniform(80, 160)
    out = eye_arr + env * env_strength
    
    # 3. Stretch/distort the environment (simulate curved glass refraction)
    # Shift rows randomly to simulate lens distortion
    out_img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    shift_amount = random.randint(2, 8)
    for row_start in range(0, h, random.randint(4, 12)):
        row_end = min(row_start + random.randint(4, 20), h)
        shift = random.randint(-shift_amount, shift_amount)
        band = out_img.crop((0, row_start, w, row_end))
        out_img.paste(band, (shift, row_start))
    out = np.array(out_img, np.float32)
    
    # 4. Specular glint spots (bright, elongated, random positions across full image)
    for _ in range(random.randint(4, 8)):
        sx = random.uniform(0.05, 0.95) * w
        sy = random.uniform(0.05, 0.95) * h
        sr_x = random.uniform(15, 60)  # horizontal radius (large)
        sr_y = random.uniform(8, 30)  # vertical radius (large)
        angle = random.uniform(0, math.pi)
        strength = random.uniform(120, 255)
        
        dx = xx - sx
        dy = yy - sy
        # Rotated ellipse
        rx = dx * math.cos(angle) + dy * math.sin(angle)
        ry = -dx * math.sin(angle) + dy * math.cos(angle)
        spot = np.exp(-(rx**2 / (2 * sr_x**2) + ry**2 / (2 * sr_y**2)))
        out = out + spot * strength
    
    # Rectangular patches with soft edges
    for _ in range(random.randint(3, 7)):
        rw = random.uniform(20, 80)
        rh = random.uniform(8, 35)
        rx0 = random.uniform(0.05, 0.85) * w
        ry0 = random.uniform(0.05, 0.85) * h
        edge_soft = random.uniform(3, 10)
        rect = np.ones((h, w), np.float32)
        rect = rect * np.exp(-((xx - rx0)**2) / (2 * edge_soft**2))
        rect = rect * np.exp(-((xx - rx0 - rw)**2) / (2 * edge_soft**2))
        rect = rect * np.exp(-((yy - ry0)**2) / (2 * edge_soft**2))
        rect = rect * np.exp(-((yy - ry0 - rh)**2) / (2 * edge_soft**2))
        out = out + rect * random.uniform(80, 180)
    
    # 5. Bright streaks (horizontal light bands from dashboard/screen)
    for _ in range(random.randint(1, 3)):
        sy = random.uniform(0.15, 0.95) * h
        band_h = random.uniform(8, 30)
        band_strength = random.uniform(60, 160)
        x_start = random.uniform(0.0, 0.3) * w
        x_end = random.uniform(0.7, 1.0) * w
        streak = np.exp(-((yy - sy) ** 2) / (2 * band_h ** 2))
        x_mask = np.clip((np.minimum(xx, x_end) - x_start) / max(1, x_end - x_start), 0, 1)
        x_mask *= np.clip((w - xx) / max(1, w - x_end), 0, 1)  # fade at edges
        out = out + streak * x_mask * band_strength
    
    # 6. Local detail washout (blur only in bright areas to hide eye detail)
    bright_mask = np.clip((out - 200) / 55, 0, 1)  # areas above 200 get blurred
    if bright_mask.max() > 0.05:
        blur_img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=random.uniform(1.5, 3.0)))
        blur_arr = np.array(blur_img, np.float32)
        out = out * (1 - bright_mask * 0.5) + blur_arr * (bright_mask * 0.5)
    
    # 7. Final bloom
    out_img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    bloom1 = np.array(out_img.filter(ImageFilter.GaussianBlur(radius=2)), np.float32)
    bloom2 = np.array(out_img.filter(ImageFilter.GaussianBlur(radius=6)), np.float32)
    out = out * 0.80 + bloom1 * 0.12 + bloom2 * 0.08
    
    return np.clip(out, 0, 255).astype(np.uint8)

for fname in samples:
    base = os.path.splitext(fname)[0]
    eye = np.array(Image.open(os.path.join(SRC_DIR, fname)).convert('L'), np.float32)
    
    for v in range(VARIANTS):
        random.seed(42 + v * 1000 + hash(fname) % 10000)
        result = synthesize(eye)
        out_path = os.path.join(OUT_DIR, f'{base}_proc_v{v}.png')
        Image.fromarray(result).save(out_path)
        print(f'{fname} v{v}')

# Grid
cell, gap, top = 128, 6, 18
cols, rows = 5, len(samples)
canvas = Image.new('L', (cols*cell+(cols-1)*gap, rows*cell+(rows-1)*gap+top), 18)
d = ImageDraw.Draw(canvas)
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
except: font = ImageFont.load_default()
for c, t in enumerate(['orig','proc-1','proc-2','proc-3','proc-4']):
    d.text((c*(cell+gap)+4, 2), t, fill=220, font=font)
for r, fname in enumerate(samples):
    base = os.path.splitext(fname)[0]; y = top + r*(cell+gap)
    canvas.paste(Image.open(os.path.join(SRC_DIR, fname)).convert('L').resize((cell, cell), Image.NEAREST), (0, y))
    for v in range(4):
        im = Image.open(os.path.join(OUT_DIR, f'{base}_proc_v{v}.png')).convert('L').resize((cell, cell), Image.NEAREST)
        canvas.paste(im, ((v+1)*(cell+gap), y))
GRID = '/home/bobby/OCEC/procedural_grid.png'
canvas.save(GRID)
print(f'Grid: {GRID}')
