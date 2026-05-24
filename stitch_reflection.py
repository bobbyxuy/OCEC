#!/usr/bin/env python3
"""Generate heavy reflection variants and stitch original + variants into one image."""
import os, sys
sys.path.insert(0, '/home/bobby/OCEC')
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_one(img, env_img=None):
    """Inline the heavy reflection generation."""
    w, h = img.size
    # env reflection
    if env_img is None:
        env = img.convert('L')
    else:
        env = env_img.convert('L')
    if env.size != (w, h):
        env = env.resize((w, h), Image.BILINEAR)
    env = env.transpose(Image.FLIP_LEFT_RIGHT)
    import random
    arr = np.array(env, dtype=np.float32)
    shift = random.randint(-h // 4, h // 4)
    arr = np.roll(arr, shift, axis=0)
    env = Image.fromarray(arr.astype(np.uint8))
    if env.mode != 'L':
        env = env.convert('L')
    env = env.transform((w, h), Image.Transform.PERSPECTIVE,
                         (random.uniform(-0.1, 0.1), 0, 0, 0, 1, 0, 0, 0, 1),
                         Image.BICUBIC, fillcolor=0)
    from PIL import ImageFilter
    env = env.filter(ImageFilter.GaussianBlur(radius=random.randint(1, 2)))
    env_arr = np.array(env, dtype=np.float32)
    env_arr = np.clip(env_arr * random.uniform(1.0, 1.5), 0, 255)
    img_arr = np.array(img, dtype=np.float32)
    intensity = random.uniform(0.5, 0.8)
    result = np.clip(img_arr * (1 - intensity) + env_arr * intensity, 0, 255).astype(np.uint8)
    result = Image.fromarray(result)
    # white spots
    overlay = np.zeros((h, w), dtype=np.float32)
    for _ in range(random.randint(3, 8)):
        cx = random.randint(w // 6, 5 * w // 6)
        cy = random.randint(h // 8, 5 * h // 8)
        r = random.randint(max(3, w // 10), max(6, w // 5))
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        dist = np.sqrt(x*x + y*y) / r
        spot = np.clip(1.0 - dist, 0, 1) ** 0.8 * random.uniform(0.7, 1.0) * 255
        overlay = np.maximum(overlay, spot)
    for _ in range(random.randint(1, 3)):
        cx = random.randint(w // 5, 4 * w // 5)
        cy = random.randint(h // 5, 3 * h // 5)
        r = random.randint(max(2, w // 20), max(4, w // 12))
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        dist = np.sqrt(x*x + y*y) / r
        spot = np.clip(1.0 - dist, 0, 1) * 255
        overlay = np.maximum(overlay, spot)
    result_arr = np.array(result, dtype=np.float32)
    result_arr = np.clip(result_arr + overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(result_arr)

import random
random.seed(42)
np.random.seed(42)

# Load env images
env_dir = '/home/bobby/OCEC/env_reflection_textures'
env_imgs = []
for f in os.listdir(env_dir):
    if f.endswith('.jpg'):
        env_imgs.append(Image.open(os.path.join(env_dir, f)))

# Pick 4 samples (2 awake, 2 sleepy)
samples = [
    'awake_0_s0036_05832_1_1_1_0_0_01.png',
    'sleepy_0_s0037_04771_1_1_0_1_0_01.png',
    'awake_2_s0035_00444_0_1_1_0_1_02.png',
    'sleepy_1_s0012_02860_0_1_0_2_1_01.png',
]

cell = 128
gap = 4
out = Image.new('L', (5 * cell + 4 * gap, 4 * cell + 4 * gap), 30)
draw = ImageDraw.Draw(out)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
except:
    font = ImageFont.load_default()

for row, fname in enumerate(samples):
    img = Image.open(f'/home/bobby/OCEC/glass1_samples/{fname}').convert('L')
    if img.size == (512, 512):
        img = img.resize((cell, cell), Image.NEAREST)
    else:
        img = img.resize((cell, cell), Image.LANCZOS)
    
    # col 0: original
    x, y = 0, row * (cell + gap)
    out.paste(img, (x, y))
    draw.text((x + 2, y + 2), 'orig', fill=200, font=font)
    
    # col 1-4: 4 variants
    for col in range(1, 5):
        env = random.choice(env_imgs)
        variant = generate_one(img, env_img=env)
        x = col * (cell + gap)
        out.paste(variant, (x, y))
        draw.text((x + 2, y + 2), f'v{col}', fill=200, font=font)

out.save('/home/bobby/OCEC/reflection_grid.png')
print('Saved /home/bobby/OCEC/reflection_grid.png', out.size)
