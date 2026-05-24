#!/usr/bin/env python3
import os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

random.seed(123)
np.random.seed(123)

SRC_DIR = '/home/bobby/OCEC/glass1_samples'
ENV_DIR = '/home/bobby/OCEC/env_reflection_textures'
OUT = '/home/bobby/OCEC/reflection_v2_grid.png'

samples = [
    'awake_0_s0036_05832_1_1_1_0_0_01.png',
    'sleepy_0_s0037_04771_1_1_0_1_0_01.png',
    'awake_2_s0035_00444_0_1_1_0_1_02.png',
    'sleepy_1_s0012_02860_0_1_0_2_1_01.png',
]

envs = []
for f in sorted(os.listdir(ENV_DIR)):
    if f.lower().endswith('.jpg'):
        envs.append(Image.open(os.path.join(ENV_DIR, f)).convert('L'))


def random_lens_mask(w, h):
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    inset_x = random.randint(0, max(1, w // 10))
    inset_y = random.randint(0, max(1, h // 10))
    x0, y0 = inset_x, inset_y
    x1, y1 = w - inset_x, h - inset_y
    shape = random.choice(['roundrect', 'poly'])
    if shape == 'roundrect':
        radius = random.randint(max(2, min(w, h)//12), max(4, min(w, h)//5))
        d.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=255)
    else:
        pts = [
            (x0 + random.randint(0, w//10), y0 + random.randint(0, h//8)),
            (x1 - random.randint(0, w//10), y0 + random.randint(0, h//10)),
            (x1 - random.randint(0, w//12), y1 - random.randint(0, h//8)),
            (x0 + random.randint(0, w//12), y1 - random.randint(0, h//10)),
        ]
        d.polygon(pts, fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=max(1, min(w,h)//18)))


def make_env_layer(w, h):
    env = random.choice(envs).resize((w, h), Image.Resampling.BILINEAR)
    env = env.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    arr = np.array(env, dtype=np.float32)
    arr = np.roll(arr, random.randint(-h//5, h//5), axis=0)
    arr = np.roll(arr, random.randint(-w//6, w//6), axis=1)
    env = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    env = env.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.2, 3.5)))
    arr = np.array(env, dtype=np.float32)
    arr = np.clip((arr / 255.0) ** random.uniform(0.7, 1.1) * 255.0, 0, 255)
    return arr


def make_spec_layer(w, h):
    layer = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(layer)
    for _ in range(random.randint(2, 5)):
        mode = random.choice(['rect', 'streak', 'blob'])
        if mode == 'rect':
            rw = random.randint(max(6, w//8), max(10, w//3))
            rh = random.randint(max(4, h//10), max(8, h//4))
            x = random.randint(0, max(0, w-rw))
            y = random.randint(0, max(0, h-rh))
            d.rounded_rectangle([x, y, x+rw, y+rh], radius=random.randint(1, max(2, rh//2)), fill=random.randint(180,255))
        elif mode == 'streak':
            y = random.randint(h//8, 7*h//8)
            thickness = random.randint(2, max(3, h//10))
            x0 = random.randint(0, w//4)
            x1 = random.randint(3*w//5, w)
            d.rectangle([x0, y, x1, min(h-1, y+thickness)], fill=random.randint(170,250))
        else:
            cx = random.randint(w//6, 5*w//6)
            cy = random.randint(h//6, 5*h//6)
            r = random.randint(max(4, w//10), max(8, w//4))
            d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=random.randint(160,255))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 4.0)))
    arr = np.array(layer, dtype=np.float32)
    arr[arr > random.randint(185, 215)] = 255
    return arr


def synthesize(img):
    base = np.array(img.convert('L'), dtype=np.float32)
    h, w = base.shape
    mask = np.array(random_lens_mask(w, h), dtype=np.float32) / 255.0
    env = make_env_layer(w, h)
    spec = make_spec_layer(w, h)

    # multiplicative degradation + additive saturation
    k = random.uniform(0.45, 0.8)
    degraded = base * (1.0 - k * mask) + env * (k * mask)
    # wash out eye detail under heavy reflection
    local_blur = np.array(Image.fromarray(np.clip(degraded,0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0,2.2))), dtype=np.float32)
    blur_alpha = mask * random.uniform(0.25, 0.45)
    degraded = degraded * (1.0 - blur_alpha) + local_blur * blur_alpha
    out = degraded + spec * mask * random.uniform(0.7, 1.0)
    out = np.clip(out, 0, 255)
    return Image.fromarray(out.astype(np.uint8))

cell = 128
gap = 6
rows = len(samples)
cols = 5
canvas = Image.new('L', (cols*cell + (cols-1)*gap, rows*cell + (rows-1)*gap + 18), 20)
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
except:
    font = ImageFont.load_default()

headers = ['orig', 'v2-1', 'v2-2', 'v2-3', 'v2-4']
for c, text in enumerate(headers):
    draw.text((c*(cell+gap)+4, 2), text, fill=220, font=font)

for r, fname in enumerate(samples):
    img = Image.open(os.path.join(SRC_DIR, fname)).convert('L').resize((cell, cell), Image.Resampling.NEAREST)
    y = 18 + r * (cell + gap)
    canvas.paste(img, (0, y))
    for c in range(1, cols):
        canvas.paste(synthesize(img), (c*(cell+gap), y))

canvas.save(OUT)
print(OUT)
