#!/usr/bin/env python3
"""
NIR Eye Image Glare Generator
在近红外(NIR)眼睛图像上生成逼真的镜片反光/光斑效果。

用法:
  # 处理整个目录
  python nir_eye_glare.py --input_dir ./eyes/ --output_dir ./eyes_glare/

  # 处理单张图片
  python nir_eye_glare.py --input ./eye.png --output ./eye_glare.png

  # 自定义参数
  python nir_eye_glare.py --input_dir ./eyes/ --output_dir ./eyes_glare/ \
      --num 5                    # 每张图生成5个变体
      --intensity 0.8            # 固定光斑强度 (0-1, 默认随机)
      --mode glint               # 模式: glint(默认) / env_blend / mixed
      --seed 42                  # 固定随机种子

模式说明:
  glint      - 椭圆高斯光斑 + 矩形柔边光块 + bloom (默认)
  env_blend  - 模糊环境纹理叠加 + glint
  mixed      - 随机混合以上两种

参数:
  --input         单张输入图片路径
  --input_dir     输入图片目录 (支持 png/jpg/bmp)
  --output        单张输出路径
  --output_dir    输出目录
  --num           每张图的变体数量 (默认: 1)
  --intensity     光斑强度 0-1 (默认: 随机 0.5-1.0)
  --mode          生成模式: glint / env_blend / mixed (默认: glint)
  --env_dir       环境纹理目录 (env_blend 模式需要, 默认: ./env_textures)
  --seed          随机种子 (默认: None)
"""
import os
import argparse
import random
import math
import numpy as np
from PIL import Image, ImageFilter


def add_glint_spots(img_arr, w, h, intensity=1.0):
    """椭圆高斯光斑 (IR LED 反光点)"""
    yy, xx = np.mgrid[:h, :w]
    out = img_arr.copy()

    for _ in range(random.randint(3, 6)):
        sx = random.uniform(0.1, 0.9) * w
        sy = random.uniform(0.1, 0.9) * h
        sr_x = random.uniform(30, 80)
        sr_y = random.uniform(10, 35)
        angle = random.uniform(-0.3, 0.3)
        dx = xx - sx
        dy = yy - sy
        rx = dx * math.cos(angle) + dy * math.sin(angle)
        ry = -dx * math.sin(angle) + dy * math.cos(angle)
        spot = np.exp(-(rx**2 / (2 * sr_x**2) + ry**2 / (2 * sr_y**2)))
        out = np.clip(out + spot * random.uniform(100, 180) * intensity, 0, 255)

    return out


def add_rect_patches(img_arr, w, h, intensity=1.0):
    """矩形柔边光块 (仪表盘/屏幕反射)"""
    yy, xx = np.mgrid[:h, :w]
    out = img_arr.copy()

    for _ in range(random.randint(8, 12)):
        rw = random.uniform(80, 220)
        rh = random.uniform(40, 120)
        rx0 = random.uniform(0.0, 0.65) * w
        ry0 = random.uniform(0.0, 0.65) * h
        edge = random.uniform(1, 4)
        rect = np.ones((h, w), np.float32)
        rect = rect * np.exp(-((xx - rx0)**2) / (2 * edge**2))
        rect = rect * np.exp(-((xx - rx0 - rw)**2) / (2 * edge**2))
        rect = rect * np.exp(-((yy - ry0)**2) / (2 * edge**2))
        rect = rect * np.exp(-((yy - ry0 - rh)**2) / (2 * edge**2))
        out = np.clip(out + rect * random.uniform(110, 200) * intensity, 0, 255)

    return out


def add_bloom(img_arr, strength=0.15):
    """Bloom 后处理"""
    out_img = Image.fromarray(img_arr.astype(np.uint8))
    bloom = np.array(out_img.filter(ImageFilter.GaussianBlur(radius=4)), np.float32)
    return np.clip(img_arr * (1 - strength) + bloom * strength, 0, 255)


def add_env_blend(img_arr, env_dir, w, h, intensity=1.0):
    """模糊环境纹理叠加"""
    env_files = sorted([f for f in os.listdir(env_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    if not env_files:
        return img_arr

    env_img = Image.open(os.path.join(env_dir, random.choice(env_files))).convert('RGB')
    env_img = env_img.resize((w, h), Image.LANCZOS)
    blur_r = random.uniform(20, 40)
    env_blur = env_img.filter(ImageFilter.GaussianBlur(radius=blur_r))
    env_gray = np.array(env_blur.convert('L'), np.float32)

    img_norm = (img_arr - img_arr.min()) / max(1, img_arr.max() - img_arr.min())
    env_norm = (env_gray - env_gray.min()) / max(1, env_gray.max() - env_gray.min())
    blend = random.uniform(0.40, 0.60) * intensity
    return (img_norm * (1 - blend) + env_norm * blend) * 255.0


def generate_glare(img, mode='glint', intensity=None, env_dir=None):
    """对单张图生成反光效果"""
    if intensity is None:
        intensity = random.uniform(0.5, 1.0)

    w, h = img.size
    out = np.array(img.convert('L'), np.float32)

    if mode == 'glint':
        out = add_glint_spots(out, w, h, intensity)
        out = add_rect_patches(out, w, h, intensity)
        out = add_bloom(out)

    elif mode == 'env_blend':
        if env_dir:
            out = add_env_blend(out, env_dir, w, h, intensity)
        out = add_glint_spots(out, w, h, intensity)
        out = add_rect_patches(out, w, h, intensity)
        out = add_bloom(out)

    elif mode == 'mixed':
        if random.random() < 0.5 and env_dir:
            out = add_env_blend(out, env_dir, w, h, intensity)
        out = add_glint_spots(out, w, h, intensity)
        out = add_rect_patches(out, w, h, intensity)
        out = add_bloom(out)

    return Image.fromarray(out.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(
        description='NIR Eye Image Glare Generator - 在近红外眼睛图像上生成镜片反光效果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python nir_eye_glare.py --input_dir ./eyes/ --output_dir ./eyes_glare/

  # 每张图生成3个变体
  python nir_eye_glare.py --input_dir ./eyes/ --output_dir ./eyes_glare/ --num 3

  # 单张图片
  python nir_eye_glare.py --input eye.png --output eye_glare.png

  # 使用环境纹理叠加模式
  python nir_eye_glare.py --input_dir ./eyes/ --output_dir ./out/ --mode env_blend --env_dir ./env_textures/""")
    parser.add_argument('--input', help='单张输入图片路径')
    parser.add_argument('--input_dir', help='输入图片目录')
    parser.add_argument('--output', help='单张输出路径')
    parser.add_argument('--output_dir', help='输出目录')
    parser.add_argument('--num', type=int, default=1, help='每张图的变体数量 (默认: 1)')
    parser.add_argument('--intensity', type=float, default=None,
                        help='光斑强度 0-1 (默认: 随机 0.5-1.0)')
    parser.add_argument('--mode', default='glint', choices=['glint', 'env_blend', 'mixed'],
                        help='生成模式 (默认: glint)')
    parser.add_argument('--env_dir', default='./env_textures',
                        help='环境纹理目录 (env_blend 模式需要, 默认: ./env_textures)')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error('需要 --input 或 --input_dir')
    if not args.output and not args.output_dir:
        parser.error('需要 --output 或 --output_dir')

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.input:
        img = Image.open(args.input)
        out = generate_glare(img, mode=args.mode, intensity=args.intensity,
                             env_dir=args.env_dir if args.mode != 'glint' else None)
        out.save(args.output)
        print(f'Saved: {args.output}')
        return

    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    files = sorted([f for f in os.listdir(args.input_dir)
                    if os.path.splitext(f)[1].lower() in exts])
    print(f'Found {len(files)} images, mode={args.mode}, variants={args.num}')

    os.makedirs(args.output_dir, exist_ok=True)
    count = 0
    for fname in files:
        img = Image.open(os.path.join(args.input_dir, fname))
        for i in range(args.num):
            if args.num == 1:
                out_name = fname
            else:
                base, ext = os.path.splitext(fname)
                out_name = f'{base}_glare{i}{ext}'
            out = generate_glare(img, mode=args.mode, intensity=args.intensity,
                                 env_dir=args.env_dir if args.mode != 'glint' else None)
            out.save(os.path.join(args.output_dir, out_name))
            count += 1
    print(f'Generated {count} images -> {args.output_dir}')


if __name__ == '__main__':
    main()
