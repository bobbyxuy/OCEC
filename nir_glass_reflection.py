#!/usr/bin/env python3
"""
NIR Glass Reflection Generator
=============================
Generate realistic near-infrared eyeglass reflection augmentation for eye state classification.

Usage:
    # Single image
    python nir_glass_reflection.py -i /path/to/eye.png -o /path/to/output.png

    # Batch directory
    python nir_glass_reflection.py -i /path/to/eyes_dir/ -o /path/to/output_dir/

    # Custom number of variants per image
    python nir_glass_reflection.py -i /path/to/eye.png -o /path/to/output.png -n 8

    # Custom environment photos directory
    python nir_glass_reflection.py -i /path/to/eye.png -o /path/to/output.png -e /path/to/env_photos/

Environment photos: should be natural scenery / indoor / car interior images.
The script will randomly sample and blend them with eye images.
"""

import os
import sys
import argparse
import random
import math
import numpy as np
from PIL import Image, ImageFilter


def get_default_env_dir():
    """Look for env photos in common locations."""
    candidates = [
        '/home/bobby/OCEC/env_reflection_textures',
        './env_reflection_textures',
        '../env_reflection_textures',
    ]
    for c in candidates:
        if os.path.isdir(c) and os.listdir(c):
            return c
    return None


def get_env_files(env_dir):
    """Get all image files from env directory."""
    if not env_dir or not os.path.isdir(env_dir):
        return []
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    return sorted([f for f in os.listdir(env_dir) if os.path.splitext(f.lower())[1] in exts])


def synthesize(eye_path, env_dir, env_files, seed, opts):
    """
    Apply NIR glass reflection to a single eye image.
    Returns a PIL Image.
    """
    eye = Image.open(eye_path).convert('L')
    ew, eh = eye.size

    rng = random.Random(seed)

    # Pick random env photo
    env_path = os.path.join(env_dir, rng.choice(env_files))
    env_img = Image.open(env_path).convert('RGB')
    env_img = env_img.resize((ew, eh), Image.LANCZOS)

    # Environment projection: heavy blur so it becomes vague shapes
    blur_r = rng.uniform(*opts['blur_r'])
    env_blur = env_img.filter(ImageFilter.GaussianBlur(radius=blur_r))
    env_gray = np.array(env_blur.convert('L'), np.float32)

    eye_arr = np.array(eye, np.float32)

    # Normalize both to [0,1]
    eye_min, eye_max = eye_arr.min(), max(1, eye_arr.max())
    env_min, env_max = env_gray.min(), max(1, env_gray.max())
    eye_norm = (eye_arr - eye_min) / (eye_max - eye_min)
    env_norm = (env_gray - env_min) / (env_max - env_min)

    # Replace eye brightness partially with env brightness
    blend = rng.uniform(*opts['blend'])
    combined = eye_norm * (1 - blend) + env_norm * blend
    out = combined * 255.0

    yy, xx = np.mgrid[:eh, :ew]

    # Ellipse glint spots
    n_ellipse = rng.randint(*opts['n_ellipse'])
    for _ in range(n_ellipse):
        sx = rng.uniform(0.05, 0.90) * ew
        sy = rng.uniform(0.05, 0.90) * eh
        sr_x = rng.uniform(*opts['ellipse_rx'])
        sr_y = rng.uniform(*opts['ellipse_ry'])
        angle = rng.uniform(-0.35, 0.35)
        strength = rng.uniform(*opts['ellipse_strength'])

        dx, dy = xx - sx, yy - sy
        rx = dx * math.cos(angle) + dy * math.sin(angle)
        ry = -dx * math.sin(angle) + dy * math.cos(angle)
        spot = np.exp(-(rx**2 / (2 * sr_x**2) + ry**2 / (2 * sr_y**2)))
        out = np.clip(out + spot * strength, 0, 255)

    # Rectangular glint patches
    n_rect = rng.randint(*opts['n_rect'])
    for _ in range(n_rect):
        rw = rng.uniform(*opts['rect_w'])
        rh = rng.uniform(*opts['rect_h'])
        rx0 = rng.uniform(0.0, max(0, 1.0 - rw / ew)) * ew
        ry0 = rng.uniform(0.0, max(0, 1.0 - rh / eh)) * eh
        edge = rng.uniform(*opts['rect_edge'])
        strength = rng.uniform(*opts['rect_strength'])

        rect = np.ones((eh, ew), np.float32)
        rect = rect * np.exp(-((xx - rx0)**2) / (2 * edge**2))
        rect = rect * np.exp(-((xx - rx0 - rw)**2) / (2 * edge**2))
        rect = rect * np.exp(-((yy - ry0)**2) / (2 * edge**2))
        rect = rect * np.exp(-((yy - ry0 - rh)**2) / (2 * edge**2))
        out = np.clip(out + rect * strength, 0, 255)

    # Bloom
    out_img = Image.fromarray(out.astype(np.uint8))
    bloom_radius = rng.uniform(*opts['bloom_r'])
    bloom = np.array(out_img.filter(ImageFilter.GaussianBlur(radius=bloom_radius)), np.float32)
    bloom_weight = rng.uniform(*opts['bloom_w'])
    out = np.clip(out * (1 - bloom_weight) + bloom * bloom_weight, 0, 255)

    return Image.fromarray(out.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(
        description='NIR Glass Reflection Generator — add realistic eyeglass reflection to NIR eye images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input eye image file or directory containing eye images')
    parser.add_argument('-o', '--output', required=True,
                        help='Output file (single image) or output directory (batch)')
    parser.add_argument('-n', '--variants', type=int, default=4,
                        help='Number of variants to generate per image (default: 4)')
    parser.add_argument('-e', '--env-dir', default=None,
                        help='Directory containing environment photos (default: auto-detect)')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--blur-r', nargs=2, type=float,
                        default=[20, 40],
                        metavar=('MIN', 'MAX'),
                        help='Environment blur radius range (default: 20 40)')
    parser.add_argument('--blend', nargs=2, type=float,
                        default=[0.40, 0.60],
                        metavar=('MIN', 'MAX'),
                        help='Environment brightness blend range (default: 0.40 0.60)')
    parser.add_argument('--n-rect', nargs=2, type=int,
                        default=[8, 12],
                        metavar=('MIN', 'MAX'),
                        help='Number of rectangular glint patches (default: 8 12)')
    parser.add_argument('--rect-w', nargs=2, type=float,
                        default=[80, 220],
                        metavar=('MIN', 'MAX'),
                        help='Rectangular patch width range in pixels (default: 80 220)')
    parser.add_argument('--rect-h', nargs=2, type=float,
                        default=[40, 120],
                        metavar=('MIN', 'MAX'),
                        help='Rectangular patch height range in pixels (default: 40 120)')
    parser.add_argument('--n-ellipse', nargs=2, type=int,
                        default=[3, 6],
                        metavar=('MIN', 'MAX'),
                        help='Number of ellipse glint spots (default: 3 6)')
    args = parser.parse_args()

    # Resolve env directory
    env_dir = args.env_dir or get_default_env_dir()
    if not env_dir:
        print('ERROR: Environment photo directory not found.', file=sys.stderr)
        print('Use -e /path/to/env_photos/ or place photos in ./env_reflection_textures/', file=sys.stderr)
        sys.exit(1)

    env_files = get_env_files(env_dir)
    if not env_files:
        print(f'ERROR: No images found in env directory: {env_dir}', file=sys.stderr)
        sys.exit(1)
    print(f'Using {len(env_files)} environment photos from: {env_dir}')

    # Build opts dict
    opts = {
        'blur_r': tuple(args.blur_r),
        'blend': tuple(args.blend),
        'n_rect': tuple(args.n_rect),
        'rect_w': tuple(args.rect_w),
        'rect_h': tuple(args.rect_h),
        'rect_edge': (1, 4),
        'rect_strength': (110, 200),
        'n_ellipse': tuple(args.n_ellipse),
        'ellipse_rx': (30, 80),
        'ellipse_ry': (10, 35),
        'ellipse_strength': (100, 180),
        'bloom_r': (3, 6),
        'bloom_w': (0.12, 0.18),
    }

    is_batch = os.path.isdir(args.input)
    base_seed = args.seed if args.seed is not None else random.randint(0, 999999)

    if is_batch:
        os.makedirs(args.output, exist_ok=True)
        input_files = []
        for f in os.listdir(args.input):
            ext = os.path.splitext(f.lower())[1]
            if ext in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.pgm', '.ppm'}:
                input_files.append(os.path.join(args.input, f))

        total = len(input_files) * args.variants
        done = 0
        print(f'Batch mode: {len(input_files)} images × {args.variants} variants = {total} outputs')

        for fname in sorted(input_files):
            base = os.path.splitext(os.path.basename(fname))[0]
            for v in range(args.variants):
                seed = base_seed + v * 10000 + hash(fname) % 100000
                result = synthesize(fname, env_dir, env_files, seed, opts)
                out_path = os.path.join(args.output, f'{base}_glare_v{v}.png')
                result.save(out_path)
                done += 1
                if done % 20 == 0 or done == total:
                    print(f'  [{done}/{total}] {os.path.basename(fname)} variant {v}')

        print(f'Done! {done} images saved to: {args.output}')

    else:
        # Single image
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        for v in range(args.variants):
            seed = base_seed + v * 10000
            result = synthesize(args.input, env_dir, env_files, seed, opts)
            if args.variants == 1:
                out_path = args.output
            else:
                base = os.path.splitext(os.path.basename(args.input))[0]
                out_path = args.output.format(base=base, variant=v)
            result.save(out_path)
            print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
