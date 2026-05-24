#!/usr/bin/env python3
"""Download car interior/dashboard images from Unsplash for use as reflection textures."""
import os
import requests
import time
from io import BytesIO
from PIL import Image

# Unsplash API (free tier, no key needed for direct URLs)
# Curated list of car dashboard/interior photo IDs from unsplash
PHOTO_IDS = [
    "ApBYWbVjEDQ", "rMFFqzLakzQ", "DqwHj015unA", "Hau7LeK4uLc",
    "0sT9M3JhJ_k", "Cn1VZKxMXGY", "rKwN7r8mBWY", "WKmP55S8R3I",
    "Vpm0HvT7U5c", "W3R_9F6sPgI", "gSgGPcQbmp0", "vnX2HnTtKNc",
    "JBMjgKx8RJo", "nTVpEK2igXQ", "o0QfP5xS5ko", "M0zWH4JUB6A",
    "5L8s0RXN3MI", "CEPab6ll3Ek", "3KMUKVFtBvU", "0oTHx1iYJjg",
    "ZmDj0p7nFxI", "8mTjSEwCXsM", "rw3cgevFxJw", "VHPOBVljJXs",
    "MsCQxIbLrJg", "dGXsNbgFJOE", "xfJ7HJ6UWQk", "Qy0KuS82qbA",
    "L7EwHr5Kq0Y", "N7MIm7QXRqk",
]

OUT_DIR = "/home/bobby/OCEC/env_reflection_textures"
os.makedirs(OUT_DIR, exist_ok=True)

success = 0
for i, pid in enumerate(PHOTO_IDS):
    url = f"https://images.unsplash.com/photo-{pid}?w=800&q=80"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content)).convert('RGB')
            # Crop to dashboard/windshield area (usually top half or center)
            w, h = img.size
            # Take center crop focusing on dashboard/windshield
            crop_h = int(h * 0.6)
            top = int(h * 0.15)
            img = img.crop((0, top, w, top + crop_h))
            # Resize to 256x256 for reflection texture
            img = img.resize((256, 256), Image.LANCZOS)
            img.save(os.path.join(OUT_DIR, f"env_{i:02d}_{pid}.jpg"))
            success += 1
            print(f"[{i+1}/{len(PHOTO_IDS)}] OK: {pid}")
        else:
            print(f"[{i+1}/{len(PHOTO_IDS)}] HTTP {r.status_code}: {pid}")
    except Exception as e:
        print(f"[{i+1}/{len(PHOTO_IDS)}] Error: {pid} - {e}")
    time.sleep(0.3)

print(f"\nDownloaded {success}/{len(PHOTO_IDS)} images to {OUT_DIR}")
