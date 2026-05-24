#!/usr/bin/env python3
"""
NIR Glasses Reflection Generator - Grayscale + Heavy Glare
生成强反光+白斑的灰度NIR图像，覆盖到看不清眼睛的程度。
"""
import os
import argparse
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def add_white_spots(img, num_spots=5, intensity=0.9):
    """在镜片区域添加多个过曝白斑（IR LED反射）"""
    w, h = img.size
    overlay = np.zeros((h, w), dtype=np.float32)
    
    for _ in range(num_spots):
        cx = random.randint(w // 6, 5 * w // 6)
        cy = random.randint(h // 8, 5 * h // 8)
        r = random.randint(max(3, w // 10), max(6, w // 5))
        
        # 多层光晕
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        dist = np.sqrt(x*x + y*y) / r
        spot = np.clip(1.0 - dist, 0, 1) ** 0.8 * intensity * 255
        overlay = np.maximum(overlay, spot)
    
    # 核心过曝
    for _ in range(random.randint(1, 3)):
        cx = random.randint(w // 5, 4 * w // 5)
        cy = random.randint(h // 5, 3 * h // 5)
        r = random.randint(max(2, w // 20), max(4, w // 12))
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        dist = np.sqrt(x*x + y*y) / r
        spot = np.clip(1.0 - dist, 0, 1) * 255
        overlay = np.maximum(overlay, spot)
    
    img_arr = np.array(img, dtype=np.float32)
    result = np.clip(img_arr + overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def add_env_reflection_grayscale(img, env_img=None, intensity=0.7, blur_sigma=2.0):
    """灰度环境反射叠加"""
    w, h = img.size
    
    if env_img is None:
        env = img.convert('L')
    else:
        env = env_img.convert('L')
    
    if env.size != (w, h):
        env = env.resize((w, h), Image.BILINEAR)
    
    # 水平翻转 + 随机偏移
    env = env.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.array(env, dtype=np.float32)
    shift = random.randint(-h // 4, h // 4)
    arr = np.roll(arr, shift, axis=0)
    env = Image.fromarray(arr.astype(np.uint8))
    
    # 确保灰度
    if env.mode != 'L':
        env = env.convert('L')
    
    # 透视变形
    env = env.transform((w, h), Image.Transform.PERSPECTIVE,
                         (random.uniform(-0.1, 0.1), 0, 0,
                          0, 1, 0,
                          0, 0, 1),
                         Image.BICUBIC, fillcolor=0)
    
    # 高斯模糊
    env = env.filter(ImageFilter.GaussianBlur(radius=max(1, int(blur_sigma))))
    
    # 亮度增强（反射在NIR中通常偏亮）
    env_arr = np.array(env, dtype=np.float32)
    env_arr = np.clip(env_arr * random.uniform(1.0, 1.5), 0, 255)
    
    # 镜片mask - 矩形/圆角矩形（方框眼镜）
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    margin = random.randint(1, max(2, w // 15))
    x0, y0 = margin, margin
    x1, y1 = w - margin, h - margin
    # 圆角矩形
    radius = random.randint(3, max(4, min(w, h) // 6))
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=220)
    # 边缘柔和
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    
    # 叠加
    img_arr = np.array(img, dtype=np.float32)
    mask_arr = np.array(mask, dtype=np.float32) / 255.0
    alpha = mask_arr * intensity
    result = np.clip(img_arr * (1 - alpha) + env_arr * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def generate_heavy_reflection(img, env_img=None):
    """生成重度反射：环境反射 + 多个白斑"""
    # 先加环境反射（强）
    result = add_env_reflection_grayscale(
        img, env_img=env_img,
        intensity=random.uniform(0.5, 0.8),
        blur_sigma=random.uniform(1.0, 2.5)
    )
    # 再加白斑（IR LED过曝）
    result = add_white_spots(
        result,
        num_spots=random.randint(3, 8),
        intensity=random.uniform(0.7, 1.0)
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--env_dir', default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    files = sorted([f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in exts])

    env_imgs = []
    if args.env_dir and os.path.isdir(args.env_dir):
        for f in os.listdir(args.env_dir):
            if os.path.splitext(f)[1].lower() in exts:
                env_imgs.append(Image.open(os.path.join(args.env_dir, f)))
        print(f"Loaded {len(env_imgs)} environment images")

    os.makedirs(args.output_dir, exist_ok=True)
    count = 0
    for fname in files:
        img = Image.open(os.path.join(args.input_dir, fname)).convert('L')

        for i in range(args.num):
            env = random.choice(env_imgs) if env_imgs else None
            result = generate_heavy_reflection(img, env_img=env)

            if args.num == 1:
                out_name = fname
            else:
                base, ext = os.path.splitext(fname)
                out_name = f"{base}_refl{i}{ext}"

            result.save(os.path.join(args.output_dir, out_name))
            count += 1

    print(f"Generated {count} images -> {args.output_dir}")


if __name__ == '__main__':
    main()
