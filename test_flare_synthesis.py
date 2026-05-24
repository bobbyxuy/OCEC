#!/usr/bin/env python3
"""
眩光合成效果对比测试

比较三种情况：
  A. 新方案 (gamma-correct linear domain synthesis)
  B. 现有方案 (PIL-based nir_glare_generator)
  C. 真实眩光样本 (glass1_samples)

输出：
  output/comparison/   - 干净/A/B/真实 四图并排
  output/histograms/   - 直方图对比
  output/stats.txt     - 关键统计量对比
"""

import os
import sys
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(__file__))
from nir_glare_generator import generate_glare

random.seed(42)
np.random.seed(42)

CLEAN_DIRS = [
    '/home/bobby/eye_datasets/data/train/awake',
    '/home/bobby/eye_datasets/data/train/sleepy',
]
REAL_FLARE_DIR = '/home/bobby/OCEC/glass1_samples'
OUT_DIR = '/home/bobby/OCEC/output'
N_SAMPLES = 100


# ──────────────────────────────────────────────
# 方案 A：gamma-correct linear domain synthesis
# ──────────────────────────────────────────────

def gen_uniform_flare(shape):
    intensity = np.random.uniform(0.2, 0.5)
    return np.full(shape, intensity, dtype=np.float32)


def gen_gaussian_flare(shape, n_blobs=None):
    h, w = shape
    n_blobs = n_blobs or np.random.randint(1, 4)
    field = np.zeros(shape, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n_blobs):
        cx = np.random.uniform(-0.2, 1.2) * w
        cy = np.random.uniform(-0.2, 1.2) * h
        sigma = np.random.uniform(0.2, 0.5) * max(h, w)
        intensity = np.random.uniform(0.5, 2.0)
        blob = intensity * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        field += blob.astype(np.float32)
    return field


def gen_gradient_flare(shape):
    h, w = shape
    direction = np.random.choice(['left', 'right', 'top', 'bottom'])
    intensity = np.random.uniform(0.3, 1.2)
    power = np.random.uniform(1.0, 2.5)
    if direction == 'left':
        ramp = np.linspace(1, 0, w) ** power
        field = np.tile(ramp, (h, 1))
    elif direction == 'right':
        ramp = np.linspace(0, 1, w) ** power
        field = np.tile(ramp, (h, 1))
    elif direction == 'top':
        ramp = np.linspace(1, 0, h) ** power
        field = np.tile(ramp.reshape(-1, 1), (1, w))
    else:
        ramp = np.linspace(0, 1, h) ** power
        field = np.tile(ramp.reshape(-1, 1), (1, w))
    return (field * intensity).astype(np.float32)


def gen_speckle_flare(shape):
    h, w = shape
    noise = np.random.rand(max(1, h // 8), max(1, w // 8)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(h, w) * 0.05)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    intensity = np.random.uniform(0.3, 1.0)
    return (noise * intensity).astype(np.float32)


def add_bloom(gray_uint8, threshold=200, blur_sigma=4, strength=0.3):
    img = gray_uint8.astype(np.float32) / 255.0
    t = threshold / 255.0
    mask = np.clip((img - t) / (1.0 - t + 1e-8), 0, 1)
    bloom = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma)
    out = img + bloom * strength
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def reduce_contrast(gray_uint8, factor=0.6):
    img = gray_uint8.astype(np.float32)
    mean = img.mean()
    out = (img - mean) * factor + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def slight_blur(gray_uint8):
    sigma = np.random.uniform(0.5, 1.5)
    return cv2.GaussianBlur(gray_uint8, (0, 0), sigmaX=sigma)


def synthesize_flare_linear(gray_uint8, flare_field, veil_strength=0.1, gamma=2.2):
    img_linear = (gray_uint8.astype(np.float32) / 255.0) ** gamma
    out_linear = img_linear + flare_field
    out_linear = out_linear * (1 - veil_strength) + veil_strength
    out = out_linear ** (1.0 / gamma)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def scheme_a(gray_uint8):
    h, w = gray_uint8.shape
    mode = np.random.choice(['uniform', 'gaussian', 'gradient', 'speckle'],
                            p=[0.3, 0.3, 0.2, 0.2])
    if mode == 'uniform':
        field = gen_uniform_flare((h, w))
    elif mode == 'gaussian':
        field = gen_gaussian_flare((h, w))
    elif mode == 'gradient':
        field = gen_gradient_flare((h, w))
    else:
        field = gen_speckle_flare((h, w))

    veil = np.random.uniform(0.05, 0.25)
    out = synthesize_flare_linear(gray_uint8, field, veil_strength=veil)

    if np.random.rand() < 0.6:
        out = add_bloom(out, threshold=200,
                        blur_sigma=np.random.uniform(2, 6),
                        strength=np.random.uniform(0.2, 0.5))
    if np.random.rand() < 0.5:
        out = reduce_contrast(out, factor=np.random.uniform(0.5, 0.85))
    if np.random.rand() < 0.3:
        out = slight_blur(out)
    return out


def scheme_b(gray_uint8):
    img_pil = Image.fromarray(gray_uint8)
    mode = random.choice(['led_spot', 'led_streak', 'ambient', 'mixed'])
    result = generate_glare(img_pil, mode=mode)
    return np.array(result.convert('L'))


# ──────────────────────────────────────────────
# 统计工具
# ──────────────────────────────────────────────

def img_stats(arr):
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'max': int(arr.max()),
        'pct_over240': float((arr > 240).mean() * 100),
        'pct_under30': float((arr < 30).mean() * 100),
    }


def collect_stats(arrays, label):
    all_stats = [img_stats(a) for a in arrays]
    keys = ['mean', 'std', 'max', 'pct_over240', 'pct_under30']
    avg = {k: np.mean([s[k] for s in all_stats]) for k in keys}
    return label, avg


# ──────────────────────────────────────────────
# 可视化工具
# ──────────────────────────────────────────────

def make_comparison_strip(clean, syn_a, syn_b, real_ref, fname, out_dir):
    """四图横排：干净 / 方案A / 方案B / 真实眩光"""
    target_h = 164  # 2x upscale for visibility
    imgs = []
    for arr in [clean, syn_a, syn_b, real_ref]:
        h, w = arr.shape
        scale = target_h / h
        resized = cv2.resize(arr, (int(w * scale), target_h), interpolation=cv2.INTER_NEAREST)
        imgs.append(resized)

    labels = ['Clean', 'Scheme A (gamma)', 'Scheme B (PIL)', 'Real Flare']
    label_h = 16
    strips = []
    for img, label in zip(imgs, labels):
        h, w = img.shape
        canvas = np.zeros((target_h + label_h, w), dtype=np.uint8)
        canvas[label_h:, :] = img
        cv2.putText(canvas, label, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 200, 1)
        strips.append(canvas)

    # Add separator lines
    sep = np.full((target_h + label_h, 2), 128, dtype=np.uint8)
    combined = strips[0]
    for s in strips[1:]:
        combined = np.hstack([combined, sep, s])

    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, fname), combined)


def make_histogram_comparison(clean_list, a_list, b_list, real_list, out_path):
    """直方图对比，4条曲线叠加"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    def hist_avg(arrays, bins=64):
        counts = np.zeros(bins)
        edges = None
        for arr in arrays:
            c, e = np.histogram(arr.flatten(), bins=bins, range=(0, 255), density=True)
            counts += c
            edges = e
        return counts / len(arrays), edges

    # Left: full histogram
    ax = axes[0]
    for arrays, label, color in [
        (clean_list, 'Clean', 'gray'),
        (a_list, 'Scheme A (gamma)', 'blue'),
        (b_list, 'Scheme B (PIL)', 'green'),
        (real_list, 'Real Flare', 'red'),
    ]:
        c, e = hist_avg(arrays)
        midpoints = (e[:-1] + e[1:]) / 2
        ax.plot(midpoints, c, label=label, color=color, linewidth=1.5)
    ax.set_title('Pixel value distribution')
    ax.set_xlabel('Pixel value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: zoom on high-value region (>200)
    ax = axes[1]
    for arrays, label, color in [
        (clean_list, 'Clean', 'gray'),
        (a_list, 'Scheme A (gamma)', 'blue'),
        (b_list, 'Scheme B (PIL)', 'green'),
        (real_list, 'Real Flare', 'red'),
    ]:
        c, e = hist_avg(arrays, bins=128)
        midpoints = (e[:-1] + e[1:]) / 2
        mask = midpoints > 180
        ax.plot(midpoints[mask], c[mask], label=label, color=color, linewidth=1.5)
    ax.set_title('High-value region (>180) zoom')
    ax.set_xlabel('Pixel value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f'Saved histogram: {out_path}')


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    # 1. Load clean images
    all_clean_files = []
    for d in CLEAN_DIRS:
        for f in os.listdir(d):
            if f.lower().endswith('.png'):
                all_clean_files.append(os.path.join(d, f))
    random.shuffle(all_clean_files)
    clean_files = all_clean_files[:N_SAMPLES]
    print(f'Loaded {len(clean_files)} clean eye images')

    clean_arrays = [cv2.imread(f, 0) for f in clean_files]
    clean_arrays = [a for a in clean_arrays if a is not None]

    # 2. Load real flare reference images (downscale to match training size)
    real_files = [os.path.join(REAL_FLARE_DIR, f)
                  for f in os.listdir(REAL_FLARE_DIR) if f.lower().endswith('.png')]
    print(f'Loaded {len(real_files)} real flare reference images')
    target_size = clean_arrays[0].shape[0]  # e.g. 82
    real_arrays_full = [cv2.imread(f, 0) for f in real_files if cv2.imread(f, 0) is not None]
    # Center-crop to same aspect, then resize
    real_arrays = []
    for arr in real_arrays_full:
        h, w = arr.shape
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        cropped = arr[y0:y0+s, x0:x0+s]
        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
        real_arrays.append(resized)

    # 3. Synthesize
    print('Synthesizing...')
    syn_a = [scheme_a(img) for img in clean_arrays]
    syn_b = [scheme_b(img) for img in clean_arrays]
    print('Done.')

    # 4. Statistics
    stats_results = []
    for label, arrays in [
        ('Clean', clean_arrays),
        ('Scheme A (gamma-domain)', syn_a),
        ('Scheme B (PIL-based)', syn_b),
        ('Real Flare (resized)', real_arrays),
    ]:
        _, avg = collect_stats(arrays, label)
        stats_results.append((label, avg))

    os.makedirs(OUT_DIR, exist_ok=True)
    stats_path = os.path.join(OUT_DIR, 'stats.txt')
    with open(stats_path, 'w') as fp:
        header = f"{'Metric':<20} {'Clean':>12} {'Scheme A':>14} {'Scheme B':>14} {'Real Flare':>14}"
        fp.write(header + '\n')
        fp.write('-' * len(header) + '\n')
        metrics = [
            ('mean', 'Mean pixel'),
            ('std', 'Std dev'),
            ('pct_over240', '% pixels > 240'),
            ('pct_under30', '% pixels < 30'),
        ]
        for key, name in metrics:
            vals = [avg[key] for _, avg in stats_results]
            row = f"{name:<20} {vals[0]:>12.2f} {vals[1]:>14.2f} {vals[2]:>14.2f} {vals[3]:>14.2f}"
            fp.write(row + '\n')
    print(f'\nStatistics saved: {stats_path}')
    with open(stats_path) as fp:
        print(fp.read())

    # 5. Comparison strips (20 samples)
    comp_dir = os.path.join(OUT_DIR, 'comparison')
    print(f'Generating comparison strips -> {comp_dir}')
    for i in range(min(20, len(clean_arrays))):
        real_ref = real_arrays[i % len(real_arrays)]
        base = os.path.basename(clean_files[i])
        make_comparison_strip(
            clean_arrays[i], syn_a[i], syn_b[i], real_ref,
            fname=f'{i:03d}_{base}', out_dir=comp_dir
        )

    # 6. Histogram comparison
    hist_path = os.path.join(OUT_DIR, 'histograms', 'distribution_comparison.png')
    try:
        make_histogram_comparison(clean_arrays, syn_a, syn_b, real_arrays, hist_path)
    except ImportError:
        print('matplotlib not available, skipping histogram plot')

    # 7. t-SNE (optional, uses HOG features as proxy)
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        def extract_feat(arr):
            # Simple HOG-like: flatten resized image
            resized = cv2.resize(arr, (32, 32))
            return resized.flatten().astype(np.float32) / 255.0

        print('Running t-SNE...')
        n_tsne = min(50, len(clean_arrays), len(real_arrays))
        feats = np.stack(
            [extract_feat(a) for a in clean_arrays[:n_tsne]] +
            [extract_feat(a) for a in syn_a[:n_tsne]] +
            [extract_feat(a) for a in syn_b[:n_tsne]] +
            [extract_feat(a) for a in (real_arrays * ((n_tsne // len(real_arrays)) + 1))[:n_tsne]]
        )
        labels_tsne = (
            ['Clean'] * n_tsne +
            ['Scheme A'] * n_tsne +
            ['Scheme B'] * n_tsne +
            ['Real Flare'] * n_tsne
        )
        pca = PCA(n_components=50, random_state=0)
        feats_pca = pca.fit_transform(feats)
        tsne = TSNE(n_components=2, random_state=0, perplexity=15)
        xy = tsne.fit_transform(feats_pca)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {'Clean': 'gray', 'Scheme A': 'blue', 'Scheme B': 'green', 'Real Flare': 'red'}
        for lbl, color in colors.items():
            mask = np.array(labels_tsne) == lbl
            ax.scatter(xy[mask, 0], xy[mask, 1], c=color, label=lbl, alpha=0.6, s=20)
        ax.legend()
        ax.set_title('t-SNE of pixel features')
        tsne_path = os.path.join(OUT_DIR, 'tsne.png')
        plt.tight_layout()
        plt.savefig(tsne_path, dpi=120)
        plt.close()
        print(f'Saved t-SNE: {tsne_path}')
    except ImportError:
        print('sklearn/matplotlib not available, skipping t-SNE')

    print(f'\nAll outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
