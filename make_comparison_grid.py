#!/usr/bin/env python3
"""
生成眩光对比网格：左原图，右合成发白/过曝效果
模拟强环境光下 NIR 相机的整体泛白现象
"""
import os, sys, random
import numpy as np
import cv2
from PIL import Image

random.seed(7)
np.random.seed(7)

AWAKE_DIR  = '/home/bobby/eye_datasets/data/train/awake'
SLEEPY_DIR = '/home/bobby/eye_datasets/data/train/sleepy'
OUT_PATH   = '/home/bobby/OCEC/output/glare_comparison.png'

SCALE      = 4
N_PER_MODE = 4


# ── 整体泛白合成函数 ──────────────────────────────────────

def synth_uniform_bright(gray):
    """均匀亮度抬升 + 轻微模糊蒙层"""
    lift = np.random.uniform(20, 60)
    blur_sigma = np.random.uniform(1.5, 3.5)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma).astype(np.float32)
    blend = np.random.uniform(0.2, 0.5)
    out = gray.astype(np.float32) * (1 - blend) + blurred * blend + lift
    return np.clip(out, 0, 255).astype(np.uint8)


def synth_gradient_bright(gray):
    """单侧渐变泛白 + 轻微模糊"""
    h, w = gray.shape
    direction = random.choice(['left', 'right', 'top', 'bottom'])
    strength = np.random.uniform(30, 90)
    power = np.random.uniform(0.8, 2.0)

    ramp = np.linspace(1.0, 0.0, w if direction in ('left', 'right') else h) ** power
    if direction == 'right':
        ramp = ramp[::-1]
    if direction in ('left', 'right'):
        field = np.tile(ramp, (h, 1))
    else:
        if direction == 'bottom':
            ramp = ramp[::-1]
        field = np.tile(ramp.reshape(-1, 1), (1, w))

    blur_sigma = np.random.uniform(2, 5)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma).astype(np.float32)
    blend_map = field * np.random.uniform(0.3, 0.6)
    out = gray.astype(np.float32) * (1 - blend_map) + blurred * blend_map + field * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def synth_haze(gray):
    """haze: light blur + mild contrast compression + slight brightness lift"""
    blur_sigma = np.random.uniform(1.5, 3.5)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma).astype(np.float32)

    factor = np.random.uniform(0.65, 0.88)
    mean = blurred.mean()
    compressed = (blurred - mean) * factor + mean

    lift = np.random.uniform(8, 30)
    out = compressed + lift
    return np.clip(out, 0, 255).astype(np.uint8)


def synth_partial_blowout(gray):
    """局部过曝蒙层：局部发白 + 扩散模糊"""
    h, w = gray.shape

    x0 = random.randint(0, w // 3)
    y0 = random.randint(0, h // 3)
    x1 = random.randint(2 * w // 3, w)
    y1 = random.randint(2 * h // 3, h)

    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    blur_sigma = np.random.uniform(8, 18)
    mask_blur = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma)
    mask_blur /= mask_blur.max() + 1e-8

    blur_sigma2 = np.random.uniform(2, 6)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma2).astype(np.float32)
    out = gray.astype(np.float32) * (1 - mask_blur * 0.5) + blurred * (mask_blur * 0.5)
    out += mask_blur * np.random.uniform(40, 100)
    return np.clip(out, 0, 255).astype(np.uint8)


MODES = {
    'uniform bright':   (synth_uniform_bright,   'uniform bright  — overall lift'),
    'gradient bright':  (synth_gradient_bright,  'gradient bright — side backlight'),
    'haze':             (synth_haze,              'haze            — blur + contrast drop + lift'),
    'partial blowout':  (synth_partial_blowout,   'partial blowout — local overexposure'),
}


# ── 工具函数 ──────────────────────────────────────────────

def load_clean_files():
    result = []
    for d in [AWAKE_DIR, SLEEPY_DIR]:
        files = [f for f in os.listdir(d) if f.replace('.png', '').split('_')[5] == '0']
        label = 'awake' if d == AWAKE_DIR else 'sleepy'
        result.extend([(os.path.join(d, f), label) for f in files])
    random.shuffle(result)
    return result


def label_img(bgr, text, pos=(4, 18), fs=0.45):
    out = bgr.copy()
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),   3, cv2.LINE_AA)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, fs, (220, 220, 220), 1, cv2.LINE_AA)
    return out


def make_pair(orig, syn, eye_label, mode_name):
    orig_up = cv2.resize(orig, (82 * SCALE, 82 * SCALE), interpolation=cv2.INTER_NEAREST)
    syn_up  = cv2.resize(syn,  (82 * SCALE, 82 * SCALE), interpolation=cv2.INTER_NEAREST)
    h = orig_up.shape[0]

    o = cv2.cvtColor(orig_up, cv2.COLOR_GRAY2BGR)
    s = cv2.cvtColor(syn_up,  cv2.COLOR_GRAY2BGR)
    o = label_img(o, eye_label)
    s = label_img(s, 'synthesized')

    sep = np.full((h, 4, 3), 55, dtype=np.uint8)
    return np.hstack([o, sep, s])


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    files = load_clean_files()
    idx = 0

    all_rows = []
    row_w = None

    for mode_key, (fn, desc) in MODES.items():
        pairs = []
        for _ in range(N_PER_MODE):
            if idx >= len(files):
                idx = 0
            fpath, label = files[idx]; idx += 1
            gray = cv2.imread(fpath, 0)
            if gray is None:
                continue
            gray = cv2.resize(gray, (82, 82), interpolation=cv2.INTER_AREA)
            syn  = fn(gray)
            pairs.append(make_pair(gray, syn, label, mode_key))

        if not pairs:
            continue

        ph, pw = pairs[0].shape[:2]
        sep_v = np.full((ph, 8, 3), 40, dtype=np.uint8)
        content_row = pairs[0]
        for p in pairs[1:]:
            content_row = np.hstack([content_row, sep_v, p])

        rw = content_row.shape[1]
        if row_w is None:
            row_w = rw

        # Header
        header = np.full((30, rw, 3), 30, dtype=np.uint8)
        cv2.putText(header, desc, (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 195, 255), 1, cv2.LINE_AA)

        all_rows.extend([header, content_row,
                         np.full((8, rw, 3), 18, dtype=np.uint8)])

    all_rows = all_rows[:-1]  # remove last separator

    # Uniform width
    max_w = max(r.shape[1] for r in all_rows)
    padded = [np.hstack([r, np.full((r.shape[0], max_w - r.shape[1], 3), 18, dtype=np.uint8)])
              if r.shape[1] < max_w else r for r in all_rows]

    grid = np.vstack(padded)
    cv2.imwrite(OUT_PATH, grid)
    print(f'Saved: {OUT_PATH}  ({grid.shape[1]}x{grid.shape[0]})')


if __name__ == '__main__':
    main()
