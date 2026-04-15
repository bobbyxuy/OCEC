#!/usr/bin/env python3
import os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

random.seed(19)
np.random.seed(19)
SRC_DIR = '/home/bobby/OCEC/glass1_samples'
ENV_DIR = '/home/bobby/OCEC/env_reflection_textures'
OUT = '/home/bobby/OCEC/reflection_singleeye_heavy_grid.png'
samples = [
    'awake_0_s0036_05832_1_1_1_0_0_01.png',
    'sleepy_0_s0037_04771_1_1_0_1_0_01.png',
    'awake_2_s0035_00444_0_1_1_0_1_02.png',
    'sleepy_1_s0012_02860_0_1_0_2_1_01.png',
]
envs = [Image.open(os.path.join(ENV_DIR, f)).convert('L') for f in sorted(os.listdir(ENV_DIR)) if f.endswith('.jpg')]

def heavy_region(w, h):
    m = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(m)
    # Larger, stronger region to genuinely occlude eye detail
    mode = random.choice(['band', 'block', 'poly'])
    if mode == 'band':
        y = random.randint(h//10, h//3)
        hh = random.randint(max(16, h//5), max(26, h//2))
        x0 = random.randint(0, w//10)
        x1 = random.randint(3*w//4, w-1)
        d.rounded_rectangle([x0, y, x1, min(h-1, y+hh)], radius=max(4, hh//4), fill=255)
    elif mode == 'block':
        rw = random.randint(max(26, w//3), max(40, 3*w//4))
        rh = random.randint(max(24, h//3), max(40, 2*h//3))
        x = random.randint(0, max(0, w-rw))
        y = random.randint(0, max(0, h-rh))
        d.rounded_rectangle([x, y, x+rw, y+rh], radius=max(5, min(rw,rh)//6), fill=255)
    else:
        pts = []
        cx = random.randint(w//3, 2*w//3)
        cy = random.randint(h//3, 2*h//3)
        for ang in np.linspace(0, 2*np.pi, 7, endpoint=False):
            rx = random.randint(max(18, w//5), max(30, w//2))
            ry = random.randint(max(14, h//5), max(26, h//2))
            pts.append((int(cx + np.cos(ang)*rx), int(cy + np.sin(ang)*ry)))
        d.polygon(pts, fill=255)
    m = m.filter(ImageFilter.GaussianBlur(radius=random.uniform(3.0, 6.0)))
    arr = np.array(m, np.float32) / 255.0
    yy = np.linspace(0, 1, h)[:, None]
    xx = np.linspace(0, 1, w)[None, :]
    # stronger upper / side emphasis
    grad = np.clip(1.25 - 0.75*yy + random.uniform(-0.22,0.22)*(xx-0.5), 0.35, 1.0)
    return np.clip(arr * grad, 0, 1)

def env_layer(w, h):
    env = random.choice(envs).resize((w, h), Image.Resampling.BILINEAR)
    env = env.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    env = env.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.2, 2.8)))
    arr = np.array(env, np.float32)
    arr = np.roll(arr, random.randint(-h//6, h//6), axis=0)
    arr = np.roll(arr, random.randint(-w//7, w//7), axis=1)
    arr = np.clip((arr/255.0)**random.uniform(0.82, 0.98)*255.0, 0, 255)
    return arr

def spec_layer(region):
    h, w = region.shape
    img = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(img)
    for _ in range(random.randint(2, 4)):
        mode = random.choice(['band', 'rect'])
        if mode == 'band':
            y = random.randint(h//10, 3*h//5)
            hh = random.randint(max(4, h//18), max(8, h//10))
            x0 = random.randint(0, w//6)
            x1 = random.randint(w//2, w-1)
            d.rounded_rectangle([x0, y, x1, min(h-1, y+hh)], radius=max(2, hh//2), fill=random.randint(190,255))
        else:
            rw = random.randint(max(10, w//8), max(18, w//3))
            rh = random.randint(max(8, h//10), max(14, h//4))
            x = random.randint(0, max(0, w-rw))
            y = random.randint(0, max(0, h-rh))
            d.rounded_rectangle([x, y, x+rw, y+rh], radius=max(3, rh//3), fill=random.randint(200,255))
    arr = np.array(img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.2, 2.5))), np.float32)
    # saturated highlight clipping
    thr = random.randint(175, 205)
    arr[arr > thr] = 255
    return arr * region

def synth(img):
    base = np.array(img, np.float32)
    h, w = base.shape
    region = heavy_region(w, h)
    env = env_layer(w, h)
    # stronger multiplicative reflection degradation
    k = random.uniform(0.38, 0.62)
    refl = np.clip((env/255.0)**random.uniform(0.8, 0.95)*255.0, 0, 255)
    out = base * (1 - k*region) + refl * (k*region)
    # heavy local wash-out to hide eye detail
    blur = np.array(Image.fromarray(np.clip(out,0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=random.uniform(1.2, 2.4))), np.float32)
    out = out * (1 - region*0.32) + blur * (region*0.32)
    # add bright saturated speculars
    spec = spec_layer(region)
    out = np.clip(out + spec * random.uniform(0.55, 0.9), 0, 255)
    # bloom-like expansion from bright parts
    bloom = Image.fromarray(np.clip(spec,0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 4.0)))
    out = np.clip(out + np.array(bloom, np.float32) * random.uniform(0.15, 0.3), 0, 255)
    return Image.fromarray(out.astype(np.uint8))

cell, gap, top = 128, 6, 18
rows, cols = len(samples), 5
canvas = Image.new('L', (cols*cell + (cols-1)*gap, rows*cell + (rows-1)*gap + top), 18)
d = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
except:
    font = ImageFont.load_default()
for c, t in enumerate(['orig','heavy-1','heavy-2','heavy-3','heavy-4']):
    d.text((c*(cell+gap)+4, 2), t, fill=220, font=font)
for r, fname in enumerate(samples):
    img = Image.open(os.path.join(SRC_DIR, fname)).convert('L').resize((cell, cell), Image.Resampling.NEAREST)
    y = top + r*(cell+gap)
    canvas.paste(img, (0, y))
    for c in range(1, cols):
        canvas.paste(synth(img), (c*(cell+gap), y))
canvas.save(OUT)
print(OUT)
