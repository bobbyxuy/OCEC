#!/usr/bin/env python3
import os, random
from PIL import Image, ImageDraw, ImageFont

random.seed(42)
out = '/home/bobby/OCEC/glass1_samples'
os.makedirs(out, exist_ok=True)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except:
    font = ImageFont.load_default()

for cls in ['awake', 'sleepy']:
    d = '/home/bobby/eye_datasets/data/test/' + cls
    fnames = [f for f in os.listdir(d) if f.split('_')[3] == '1']
    samples = random.sample(fnames, min(8, len(fnames)))
    for i, f in enumerate(samples):
        img = Image.open(os.path.join(d, f)).convert('RGB')
        big = img.resize((512, 512), Image.NEAREST)
        draw = ImageDraw.Draw(big)
        draw.rectangle([0, 0, 512, 25], fill=(0, 0, 0))
        draw.text((10, 3), cls + ' ' + f, fill=(255, 255, 255), font=font)
        big.save(os.path.join(out, '%s_%d_%s' % (cls, i, f)))
        print(cls, f, img.size)

print('Saved to', out)
