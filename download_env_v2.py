#!/usr/bin/env python3
"""Download car interior images via Unsplash Source API."""
import os, time, requests
from io import BytesIO
from PIL import Image

OUT_DIR = "/home/bobby/OCEC/env_reflection_textures"
os.makedirs(OUT_DIR, exist_ok=True)

# Use Unsplash source API - returns redirect to actual image
queries = [
    "car dashboard", "car interior", "driving view", "car windshield",
    "car cockpit", "steering wheel view", "car cabin",
]

downloaded = 0
for q in queries:
    for attempt in range(5):
        if downloaded >= 30:
            break
        url = f"https://source.unsplash.com/800x600/?{q}"
        try:
            r = requests.get(url, timeout=15, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 10000:
                img = Image.open(BytesIO(r.content)).convert('RGB')
                w, h = img.size
                # Crop center 60%
                crop_h = int(h * 0.6)
                top = int(h * 0.15)
                img = img.crop((0, top, w, top + crop_h))
                img = img.resize((256, 256), Image.LANCZOS)
                fname = f"env_{downloaded:02d}_{q.replace(' ','_')}.jpg"
                img.save(os.path.join(OUT_DIR, fname))
                downloaded += 1
                print(f"[{downloaded}] OK: {q} ({len(r.content)//1024}KB)")
            else:
                print(f"[{downloaded}] skip: HTTP {r.status_code} or too small")
        except Exception as e:
            print(f"[{downloaded}] Error: {e}")
        time.sleep(1)

print(f"\nTotal: {downloaded} images -> {OUT_DIR}")
