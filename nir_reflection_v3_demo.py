#!/usr/bin/env python3
import os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

random.seed(7)
np.random.seed(7)
SRC_DIR = '/home/bobby/OCEC/glass1_samples'
ENV_DIR = '/home/bobby/OCEC/env_reflection_textures'
OUT = '/home/bobby/OCEC/reflection_v3_grid.png'

samples = [
    'awake_0_s0036_05832_1_1_1_0_0_01.png',
    'sleepy_0_s0037_04771_1_1_0_1_0_01.png',
    'awake_2_s0035_00444_0_1_1_0_1_02.png',
    'sleepy_1_s0012_02860_0_1_0_2_1_01.png',
]

envs = [Image.open(os.path.join(ENV_DIR, f)).convert('L') for f in sorted(os.listdir(ENV_DIR)) if f.endswith('.jpg')]


def two_lens_masks(w, h):
    left = Image.new('L', (w, h), 0)
    right = Image.new('L', (w, h), 0)
    dl, dr = ImageDraw.Draw(left), ImageDraw.Draw(right)
    lens_w = int(w * random.uniform(0.34, 0.42))
    lens_h = int(h * random.uniform(0.58, 0.72))
    gap = int(w * random.uniform(0.04, 0.1))
    cy = int(h * random.uniform(0.48, 0.54))
    lx1 = w // 2 - gap // 2
    lx0 = lx1 - lens_w
    rx0 = w // 2 + gap // 2
    rx1 = rx0 + lens_w
    y0 = cy - lens_h // 2
    y1 = cy + lens_h // 2
    radius = max(3, int(min(lens_w, lens_h) * random.uniform(0.12, 0.22)))
    dl.rounded_rectangle([lx0, y0, lx1, y1], radius=radius, fill=255)
    dr.rounded_rectangle([rx0, y0, rx1, y1], radius=radius, fill=255)
    left = left.filter(ImageFilter.GaussianBlur(radius=1.2))
    right = right.filter(ImageFilter.GaussianBlur(radius=1.2))
    outline = Image.new('L', (w, h), 0)
    do = ImageDraw.Draw(outline)
    do.rounded_rectangle([lx0, y0, lx1, y1], radius=radius, outline=180, width=max(1, w // 64))
    do.rounded_rectangle([rx0, y0, rx1, y1], radius=radius, outline=180, width=max(1, w // 64))
    return np.array(left, np.float32) / 255.0, np.array(right, np.float32) / 255.0, np.array(outline, np.float32)


def env_patch(w, h):
    env = random.choice(envs).resize((w, h), Image.Resampling.BILINEAR)
    env = env.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    env = env.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.2)))
    arr = np.array(env, np.float32)
    arr = np.roll(arr, random.randint(-h // 8, h // 8), axis=0)
    arr = np.roll(arr, random.randint(-w // 8, w // 8), axis=1)
    arr = np.clip((arr / 255.0) ** random.uniform(0.85, 1.05) * 255.0, 0, 255)
    return arr


def local_reflection_mask(mask):
    h, w = mask.shape
    m = Image.fromarray((mask * 255).astype(np.uint8))
    draw = ImageDraw.Draw(m)
    # Keep reflection mainly upper/side area inside lens, not full lens
    for _ in range(random.randint(1, 2)):
        rw = random.randint(max(8, w // 8), max(16, w // 3))
        rh = random.randint(max(6, h // 10), max(12, h // 4))
        x = random.randint(0, max(0, w - rw))
        y = random.randint(0, max(0, h // 2))
        draw.rounded_rectangle([x, y, x + rw, y + rh], radius=max(2, rh // 4), fill=255)
    arr = np.array(m.filter(ImageFilter.GaussianBlur(radius=4)), np.float32) / 255.0
    return np.clip(arr * mask, 0, 1)


def specular_layer(mask):
    h, w = mask.shape
    img = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(img)
    for _ in range(random.randint(1, 3)):
        mode = random.choice(['streak', 'rect'])
        if mode == 'streak':
            y = random.randint(h // 6, h // 2)
            hh = random.randint(2, max(3, h // 14))
            x0 = random.randint(0, w // 3)
            x1 = random.randint(2 * w // 3, w - 1)
            d.rounded_rectangle([x0, y, x1, y + hh], radius=max(1, hh // 2), fill=random.randint(170, 235))
        else:
            rw = random.randint(max(6, w // 10), max(10, w // 5))
            rh = random.randint(max(5, h // 12), max(8, h // 6))
            x = random.randint(0, max(0, w - rw))
            y = random.randint(0, max(0, h // 2))
            d.rounded_rectangle([x, y, x + rw, y + rh], radius=max(2, rh // 3), fill=random.randint(175, 245))
    arr = np.array(img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.0))), np.float32)
    return arr * mask


def synth(img):
    base = np.array(img, np.float32)
    h, w = base.shape
    lmask, rmask, outline = two_lens_masks(w, h)
    env = env_patch(w, h)
    lref = local_reflection_mask(lmask)
    rref = local_reflection_mask(rmask)
    ref = np.clip(lref + rref, 0, 1)
    # mild multiplicative degradation, preserve eye visibility
    k = random.uniform(0.18, 0.34)
    out = base * (1 - ref * k) + env * (ref * k)
    blur = np.array(Image.fromarray(np.clip(out,0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2))), np.float32)
    out = out * (1 - ref * 0.12) + blur * (ref * 0.12)
    spec = specular_layer(ref)
    out = np.clip(out + spec * random.uniform(0.35, 0.6), 0, 255)
    # lens edge hint
    out = np.clip(out + outline * 0.18, 0, 255)
    return Image.fromarray(out.astype(np.uint8))

cell = 128
rows, cols, gap, top = len(samples), 5, 6, 18
canvas = Image.new('L', (cols * cell + (cols - 1) * gap, rows * cell + (rows - 1) * gap + top), 18)
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
except:
    font = ImageFont.load_default()
for c, t in enumerate(['orig', 'v3-1', 'v3-2', 'v3-3', 'v3-4']):
    draw.text((c * (cell + gap) + 4, 2), t, fill=220, font=font)
for r, fname in enumerate(samples):
    img = Image.open(os.path.join(SRC_DIR, fname)).convert('L').resize((cell, cell), Image.Resampling.NEAREST)
    y = top + r * (cell + gap)
    canvas.paste(img, (0, y))
    for c in range(1, cols):
        canvas.paste(synth(img), (c * (cell + gap), y))
canvas.save(OUT)
print(OUT)
