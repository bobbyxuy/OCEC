#!/usr/bin/env python3
"""
NIR Glasses Environment Reflection Generator
在眼镜镜片上生成逼真的环境反射效果（能看到反射的环境）。

原理：
  1. 取环境源图（默认用图片自身，或指定环境图目录）
  2. 水平翻转（镜面反射）
  3. 透视变形（模拟镜片曲率）
  4. 高斯模糊（反射总是模糊的）
  5. 混合叠加到镜片区域
"""
import os
import argparse
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTransform


def create_lens_mask(w, h, mode='upper'):
    """创建镜片区域 mask（椭圆形，覆盖眼睛图上半部分）"""
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)

    if mode == 'upper':
        # 镜片在眼睛图上半部分
        cx = w // 2
        cy = int(h * 0.3)
        rx = int(w * 0.45)
        ry = int(h * 0.35)
        # 上边缘稍平（镜片形状）
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=180)
        # 渐变边缘：上部更实，下部渐隐
        for i in range(ry):
            y = cy - ry + i
            alpha = int(180 * (i / ry) ** 0.5)  # 底部渐隐
            draw.ellipse([cx - rx + i // 3, y, cx + rx - i // 3, y + 2], fill=min(255, alpha + 60))
    elif mode == 'full':
        # 全图镜片（覆盖更大区域）
        cx, cy = w // 2, h // 2
        rx, ry = int(w * 0.48), int(h * 0.45)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=200)
        # 边缘渐变
        for r in range(min(rx, ry), 0, -1):
            alpha = int(200 * (r / min(rx, ry)) ** 0.3)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=min(255, alpha))

    return mask.filter(ImageFilter.GaussianBlur(radius=3))


def perspective_warp(img, strength=0.3):
    """对图像做透视变形，模拟镜片曲率"""
    w, h = img.size
    # 水平方向压缩中间，拉伸两边（桶形畸变）
    coeffs = [
        strength * 0.1,   # x -> x (微弱非线性)
        0,
        strength * -0.05,  # x -> y
        0,
        1,
        0,
        0,
        0,
        1,
    ]
    return img.transform((w, h), Image.Transform.PERSPECTIVE, coeffs,
                          Image.BICUBIC, fillcolor=0)


def add_environment_reflection(img, env_img=None, intensity=0.35, blur_sigma=4.0):
    """
    在眼睛图像上叠加环境反射。

    参数:
        img: 输入眼睛图像 (PIL Image, 灰度或RGB)
        env_img: 环境源图 (None则用图片自身)
        intensity: 反射强度 0-1
        blur_sigma: 反射模糊程度
    """
    w, h = img.size

    # 环境源
    if env_img is None:
        env = img.copy()
    else:
        env = env_img.copy()

    # 缩放环境图到与眼睛图相同尺寸
    if env.size != (w, h):
        env = env.resize((w, h), Image.BILINEAR)

    # 水平翻转（镜面反射）
    env = env.transpose(Image.FLIP_LEFT_RIGHT)

    # 垂直偏移（反射的内容来自上方环境）
    shift = random.randint(-h // 6, h // 6)
    env_arr = np.array(env)
    env_arr = np.roll(env_arr, shift, axis=0)
    env = Image.fromarray(env_arr)

    # 透视变形（镜片曲率）
    env = perspective_warp(env, strength=random.uniform(0.2, 0.5))

    # 高斯模糊（反射总是模糊的）
    blur_r = max(2, int(blur_sigma))
    env = env.filter(ImageFilter.GaussianBlur(radius=blur_r))

    # 亮度调整（反射通常比原场景暗）
    env_arr = np.array(env, dtype=np.float32)
    brightness = random.uniform(0.7, 1.2)
    env_arr = np.clip(env_arr * brightness, 0, 255).astype(np.uint8)
    env = Image.fromarray(env_arr)

    # 创建镜片 mask
    mask = create_lens_mask(w, h, mode=random.choice(['upper', 'full']))

    # 叠加
    img_arr = np.array(img, dtype=np.float32)
    env_arr = np.array(env, dtype=np.float32)
    mask_arr = np.array(mask, dtype=np.float32) / 255.0

    # mask/env 扩展到与 img 相同通道数
    if len(img_arr.shape) == 3:
        if len(env_arr.shape) == 2:
            env_arr = np.stack([env_arr]*3, axis=-1)
        mask_arr = mask_arr[:, :, np.newaxis]
    else:
        mask_arr = mask_arr[:, :, np.newaxis]
        env_arr = env_arr[:, :, np.newaxis]

    alpha = mask_arr * intensity
    result = np.clip(
        img_arr * (1 - alpha) + env_arr * alpha,
        0, 255
    ).astype(np.uint8)

    return Image.fromarray(result)


def generate_reflection(img, env_img=None, intensity=None):
    """生成单张环境反射"""
    if intensity is None:
        intensity = random.uniform(0.25, 0.5)
    base = intensity if intensity else 0.4
    blur = random.uniform(1.0, 3.0)  # less blur for stronger reflection
    return add_environment_reflection(img, env_img=env_img, intensity=intensity, blur_sigma=blur)


def main():
    parser = argparse.ArgumentParser(description='NIR Glasses Environment Reflection Generator')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num', type=int, default=1, help='Variants per image')
    parser.add_argument('--env_dir', default=None, help='Environment source images (default: self)')
    parser.add_argument('--intensity', type=float, default=None, help='Fixed intensity 0-1')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    files = sorted([f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in exts])

    # 加载环境图
    env_imgs = []
    if args.env_dir and os.path.isdir(args.env_dir):
        for f in os.listdir(args.env_dir):
            if os.path.splitext(f)[1].lower() in exts:
                env_imgs.append(Image.open(os.path.join(args.env_dir, f)))
        print(f"Loaded {len(env_imgs)} environment images")

    os.makedirs(args.output_dir, exist_ok=True)
    count = 0
    for fname in files:
        img = Image.open(os.path.join(args.input_dir, fname))

        for i in range(args.num):
            env = random.choice(env_imgs) if env_imgs else None
            result = generate_reflection(img, env_img=env, intensity=args.intensity)

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
