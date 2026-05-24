#!/usr/bin/env python3
"""
NIR Glasses Glare Generator
在眼睛图像上生成逼真的近红外眼镜反光效果。

Usage:
  python nir_glare_generator.py --input_dir ./eyes/ --output_dir ./eyes_glare/ --num 3

Modes:
  led_spot     - 单个/双个 IR LED 高亮圆斑（最常见）
  led_streak   - IR LED 拉丝反射（曲面镜片）
  ambient      - 环境光漫反射（大面积弱高光）
  mixed        - 随机混合以上模式
"""
import os
import argparse
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def add_led_spot(img, intensity=1.0, num_spots=2):
    """单个或双个 IR LED 圆斑反光（最常见模式）"""
    w, h = img.size
    overlay = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(overlay)

    # 反光位置：镜片上半部分（眼睛图的上 1/3 区域）
    for _ in range(num_spots):
        cx = random.randint(w // 5, 4 * w // 5)
        cy = random.randint(h // 8, h // 3)
        # IR LED 反光大小：小而亮
        radius_x = random.randint(max(3, w // 12), max(5, w // 6))
        radius_y = random.randint(max(2, h // 12), max(4, h // 8))

        # 外层光晕（柔和扩散）
        for r_mult, alpha in [(3.0, 0.15), (2.0, 0.3), (1.5, 0.5), (1.0, 0.9)]:
            rx, ry = int(radius_x * r_mult), int(radius_y * r_mult)
            brightness = int(255 * alpha * intensity)
            draw.ellipse(
                [cx - rx, cy - ry, cx + rx, cy + ry],
                fill=brightness
            )

        # 核心过曝点（像素饱和，纯白）
        core_r = max(1, radius_x // 4)
        draw.ellipse(
            [cx - core_r, cy - core_r, cx + core_r, cy + core_r],
            fill=255
        )

    # 轻微高斯模糊使边缘自然
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.5))

    # 合成
    if img.mode == 'L':
        img_np = np.array(img, dtype=np.float32)
        ov_np = np.array(overlay, dtype=np.float32)
        result = np.clip(img_np + ov_np * 0.7, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
    else:
        # RGB: 叠加白色高光
        img_np = np.array(img, dtype=np.float32)
        ov_np = np.array(overlay, dtype=np.float32) / 255.0
        for c in range(3):
            img_np[:, :, c] = np.clip(
                img_np[:, :, c] * (1 - ov_np * 0.6) + 255 * ov_np * 0.6,
                0, 255
            )
        return Image.fromarray(img_np.astype(np.uint8))


def add_led_streak(img, intensity=1.0):
    """IR LED 拉丝/条状反光（曲面镜片上常见）"""
    w, h = img.size
    overlay = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(overlay)

    # 水平或略倾斜的条状高光
    angle = random.uniform(-15, 15)
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(h // 6, h // 3)
    streak_len = random.randint(w // 3, w // 2)
    streak_w = random.randint(max(2, h // 20), max(3, h // 12))

    # 画拉丝：多个重叠椭圆沿一条线
    steps = random.randint(5, 12)
    for i in range(steps):
        t = i / (steps - 1) - 0.5
        sx = cx + int(t * streak_len)
        sy = cy + int(t * streak_len * np.tan(np.radians(angle)))
        rx = random.randint(streak_w, streak_w * 2)
        ry = random.randint(max(1, streak_w // 2), streak_w)
        alpha = int(200 * intensity * (1 - abs(t) * 0.5))
        draw.ellipse([sx - rx, sy - ry, sx + rx, sy + ry], fill=alpha)

    # 中心最亮
    core_r = max(2, streak_w // 2)
    draw.ellipse([cx - core_r * 2, cy - core_r, cx + core_r * 2, cy + core_r], fill=int(255 * intensity))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.0))

    if img.mode == 'L':
        img_np = np.array(img, dtype=np.float32)
        ov_np = np.array(overlay, dtype=np.float32)
        result = np.clip(img_np + ov_np * 0.5, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
    else:
        img_np = np.array(img, dtype=np.float32)
        ov_np = np.array(overlay, dtype=np.float32) / 255.0
        for c in range(3):
            img_np[:, :, c] = np.clip(
                img_np[:, :, c] * (1 - ov_np * 0.5) + 255 * ov_np * 0.5,
                0, 255
            )
        return Image.fromarray(img_np.astype(np.uint8))


def add_ambient_glare(img, intensity=1.0):
    """环境光漫反射（大面积弱高光，镜片边缘）"""
    w, h = img.size
    overlay = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(overlay)

    # 在镜片上方的弧形区域画渐变高光
    y_base = random.randint(h // 6, h // 3)
    for i in range(h // 3):
        y = y_base - i
        if y < 0:
            break
        alpha = int(80 * intensity * (1 - i / (h / 3)))
        x_center = w // 2 + random.randint(-w // 8, w // 8)
        x_half = int(w * 0.4 * (1 - i / (h / 3) * 0.3))
        draw.ellipse(
            [x_center - x_half, y - 2, x_center + x_half, y + 2],
            fill=alpha
        )

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3))

    if img.mode == 'L':
        img_np = np.array(img, dtype=np.float32)
        ov_np = np.array(overlay, dtype=np.float32)
        result = np.clip(img_np + ov_np * 0.3, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
    else:
        img_np = np.array(img, dtype=np.float32)
        ov_np = np.array(overlay, dtype=np.float32) / 255.0
        for c in range(3):
            img_np[:, :, c] = np.clip(
                img_np[:, :, c] * (1 - ov_np * 0.3) + 255 * ov_np * 0.3,
                0, 255
            )
        return Image.fromarray(img_np.astype(np.uint8))


def add_mixed(img, intensity=1.0):
    """随机组合多种反光模式"""
    funcs = [add_led_spot, add_led_streak, add_ambient_glare]
    # 至少 1 种，最多 2 种
    chosen = random.sample(funcs, k=random.randint(1, 2))
    result = img
    for f in chosen:
        sub_intensity = intensity * random.uniform(0.5, 1.0)
        result = f(result, intensity=sub_intensity)
    return result


def generate_glare(img, mode='mixed', intensity=None):
    """对单张图生成反光"""
    if intensity is None:
        intensity = random.uniform(0.4, 1.0)

    modes = {
        'led_spot': add_led_spot,
        'led_streak': add_led_streak,
        'ambient': add_ambient_glare,
        'mixed': add_mixed,
    }
    return modes[mode](img, intensity=intensity)


def main():
    parser = argparse.ArgumentParser(description='NIR Glasses Glare Generator')
    parser.add_argument('--input_dir', required=True, help='Input eye images')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--num', type=int, default=1, help='Number of glare variants per image (default: 1)')
    parser.add_argument('--mode', default='mixed', choices=['led_spot', 'led_streak', 'ambient', 'mixed'],
                        help='Glare mode (default: mixed)')
    parser.add_argument('--intensity', type=float, default=None,
                        help='Fixed intensity 0-1 (default: random 0.4-1.0)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    files = sorted([f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in exts])
    print(f"Found {len(files)} images, generating {args.num} variant(s) each")

    count = 0
    for fname in files:
        img = Image.open(os.path.join(args.input_dir, fname))

        for i in range(args.num):
            glare_img = generate_glare(img, mode=args.mode, intensity=args.intensity)
            if args.num == 1:
                out_name = fname
            else:
                base, ext = os.path.splitext(fname)
                out_name = f"{base}_glare{i}{ext}"

            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, out_name)
            glare_img.save(out_path)
            count += 1

    print(f"Generated {count} images -> {args.output_dir}")


if __name__ == '__main__':
    main()
