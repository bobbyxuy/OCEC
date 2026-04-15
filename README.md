# OCEC - Open Closed Eye Classifier

近红外(NIR)眼睛图像分类与数据增强工具集。

## 功能

### NIR Eye Image Glare Generator

在近红外眼睛图像上生成逼真的镜片反光/光斑效果，用于数据增强，提升模型对戴眼镜场景的鲁棒性。

**支持三种模式：**

- `glint` — 椭圆高斯光斑 + 矩形柔边光块 + bloom（默认，推荐）
- `env_blend` — 模糊环境纹理叠加 + glint（需提供环境纹理）
- `mixed` — 随机混合以上两种

### 快速开始

```bash
# 安装依赖
pip install pillow numpy

# 基本用法 - 处理整个目录
python nir_eye_glare.py --input_dir ./eyes/ --output_dir ./eyes_glare/

# 每张图生成3个变体
python nir_eye_glare.py --input_dir ./eyes/ --output_dir ./eyes_glare/ --num 3

# 处理单张图片
python nir_eye_glare.py --input eye.png --output eye_glare.png

# 自定义强度
python nir_eye_glare.py --input_dir ./eyes/ --output_dir ./eyes_glare/ --intensity 0.8

# 使用环境纹理叠加模式
python nir_eye_glare.py --input_dir ./eyes/ --output_dir ./eyes_glare/ --mode env_blend --env_dir ./env_textures/
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 单张输入图片路径 | - |
| `--input_dir` | 输入图片目录 | - |
| `--output` | 单张输出路径 | - |
| `--output_dir` | 输出目录 | - |
| `--num` | 每张图的变体数量 | 1 |
| `--intensity` | 光斑强度 0-1 | 随机 0.5-1.0 |
| `--mode` | 生成模式: glint / env_blend / mixed | glint |
| `--env_dir` | 环境纹理目录 | ./env_textures |
| `--seed` | 随机种子 | None |

## 效果说明

- **glint（默认）**: 模拟 IR LED 在镜片上的高光反射，包含椭圆光斑和矩形屏幕反射
- **env_blend**: 在 glint 基础上叠加模糊环境纹理，模拟车内环境反射

## DINOv2 眼睛睁闭分类器

基于 DINOv2-ViT-Large 微调的二分类模型（睁眼/闭眼），在 MRL Eye Dataset 上达到 F1 > 0.99。

```bash
python train_dinov2.py
```

## 数据集

使用 [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) — 84,898 张红外眼图，37 个被试者。

## 依赖

- Python >= 3.8
- PyTorch
- timm (DINOv2 backbone)
- Pillow
- numpy
- scikit-learn

## License

MIT
