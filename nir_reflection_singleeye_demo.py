#!/usr/bin/env python3
import os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

random.seed(11)
np.random.seed(11)
SRC_DIR = '/home/bobby/OCEC/glass1_samples'
ENV_DIR = '/home/bobby/OCEC/env_reflection_textures'
OUT = '/home/bobby/OCEC/reflection_singleeye_grid.png'
samples = [
    'awake_0_s0036_05832_1_1_1_0_0_01.png',
    'sleepy_0_s0037_04771_1_1_0_1_0_01.png',
    'awake_2_s0035_00444_0_1_1_0_1_02.png',
    'sleepy_1_s0012_02860_0_1_0_2_1_01.png',
]
envs = [Image.open(os.path.join(ENV_DIR, f)).convert('L') for f in sorted(os.listdir(ENV_DIR)) if f.endswith('.jpg')]

def reflection_region(w, h):
    m = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(m)
    mode = random.choice(['band', 'poly', 'block'])
    if mode == 'band':
        y = random.randint(h//8, h//2)
        hh = random.randint(max(6, h//12), max(10, h//4))
        x0 = random.randint(0, w//6)
        x1 = random.randint(2*w//3, w-1)
        d.rounded_rectangle([x0, y, x1, min(h-1, y+hh)], radius=max(2, hh//3), fill=255)
    elif mode == 'block':
        rw = random.randint(max(10, w//6), max(18, w//2))
        rh = random.randint(max(10, h//6), max(18, h//2))
        x = random.randint(0, max(0, w-rw))
        y = random.randint(0, max(0, h-rh))
        d.rounded_rectangle([x, y, x+rw, y+rh], radius=max(3, min(rw, rh)//5), fill=255)
    else:
        pts = []
        cx = random.randint(w//3, 2*w//3)
        cy = random.randint(h//4, 2*h//3)
        for ang in np.linspace(0, 2*np.pi, 6, endpoint=False):
            rx = random.randint(max(8, w//10), max(14, w//3))
            ry = random.randint(max(8, h//10), max(14, h//3))
            pts.append((int(cx + np.cos(ang)*rx), int(cy + np.sin(ang)*ry)))
        d.polygon(pts, fill=255)
    m = m.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 4.5)))
    arr = np.array(m, np.float32) / 255.0
    # keep mostly upper / outer side stronger
    yy = np.linspace(0, 1, h)[:, None]
    xx = np.linspace(0, 1, w)[None, :]
    grad = np.clip(1.15 - 0.85*yy + random.uniform(-0.1,0.1)*(xx-0.5), 0.2, 1.0)
    return np.clip(arr * grad, 0, 1)

def env_layer(w, h):
    env = random.choice(envs).resize((w, h), Image.Resampling.BILINEAR)
    env = env.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    env = env.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.3)))
    arr = np.array(env, np.float32)
    arr = np.roll(arr, random.randint(-h//7, h//7), axis=0)
    arr = np.roll(arr, random.randint(-w//8, w//8), axis=1)
    arr = np.clip((arr/255.0)**random.uniform(0.9,1.05)*255.0, 0, 255)
    return arr

def spec_layer(region):
    h, w = region.shape
    img = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(img)
    for _ in range(random.randint(1, 2)):
        if random.random() < 0.6:
            y = random.randint(h//8, h//2)
            hh = random.randint(2, max(3, h//14))
            x0 = random.randint(0, w//5)
            x1 = random.randint(w//2, w-1)
            d.rounded_rectangle([x0, y, x1, y+hh], radius=max(1, hh//2), fill=random.randint(170, 230))
        else:
            rw = random.randint(max(6, w//12), max(10, w//5))
            rh = random.randint(max(5, h//12), max(8, h//5))
            x = random.randint(0, max(0, w-rw))
            y = random.randint(0, max(0, h-rh))
            d.rounded_rectangle([x, y, x+rw, y+rh], radius=max(2, rh//3), fill=random.randint(180, 240))
    arr = np.array(img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 1.8))), np.float32)
    return arr * region

def synth(img):
    base = np.array(img, np.float32)
    h, w = base.shape
    region = reflection_region(w, h)
    env = env_layer(w, h)
    # mild nonlinear reflection blend
    k = random.uniform(0.16, 0.3)
    refl = np.clip((env/255.0)**random.uniform(0.8, 1.0)*255.0, 0, 255)
    out = base * (1 - k*region) + refl * (k*region)
    # slight detail wash-out only in reflection region
    blur = np.array(Image.fromarray(np.clip(out,0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.0))), np.float32)
    out = out * (1 - region*0.1) + blur * (region*0.1)
    spec = spec_layer(region)
    out = np.clip(out + spec * random.uniform(0.22, 0.42), 0, 255)
    return Image.fromarray(out.astype(np.uint8))

cell, gap, top = 128, 6, 18
rows, cols = len(samples), 5
canvas = Image.new('L', (cols*cell + (cols-1)*gap, rows*cell + (rows-1)*gap + top), 18)
d = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
except:
    font = ImageFont.load_default()
for c, t in enumerate(['orig','single-1','single-2','single-3','single-4']):
    d.text((c*(cell+gap)+4, 2), t, fill=220, font=font)
for r, fname in enumerate(samples):
    img = Image.open(os.path.join(SRC_DIR, fname)).convert('L').resize((cell, cell), Image.Resampling.NEAREST)
    y = top + r*(cell+gap)
    canvas.paste(img, (0, y))
    for c in range(1, cols):
        canvas.paste(synth(img), (c*(cell+gap), y))
canvas.save(OUT)
print(OUT)
