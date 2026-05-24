# NIR Glass Reflection Generator

生成近红外眼镜反光增强数据，用于眼睛睁闭状态分类模型的难例增强。

## 效果示例

原始 NIR 眼睛图像经过处理后，叠加：
- **模糊环境投影**：真实环境照片（风景/车内）经重度模糊后叠加，隐约可见环境纹理
- **矩形高光斑**：多个大尺寸矩形软边光斑，模拟眼镜镜面反射
- **椭圆高光斑**：少量椭圆拉丝光斑，增加自然感
- **Bloom**：整体泛光，让高光区域更柔和

## 安装依赖

```bash
pip install numpy Pillow
```

## 快速开始

### 单张图像

```bash
python nir_glass_reflection.py \
    -i input/eye.png \
    -o output/eye_glare.png
```

生成 4 个变体（默认）：
```bash
python nir_glass_reflection.py -i input/eye.png -o output/eye_glare.png -n 4
```

### 批量目录

```bash
python nir_glass_reflection.py \
    -i /path/to/glass1_images/ \
    -o /path/to/output_dir/ \
    -n 4
```

输出文件名格式：`{原文件名}_glare_v{0,1,2,...}.png`

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-n VARIANTS` | 4 | 每张图生成的变体数量 |
| `--blur-r MIN MAX` | 20 40 | 环境投影模糊半径，越大越模糊 |
| `--blend MIN MAX` | 0.40 0.60 | 环境亮度替换强度，越大眼睛越像环境 |
| `--n-rect MIN MAX` | 8 12 | 矩形光斑数量 |
| `--rect-w MIN MAX` | 80 220 | 矩形宽度（像素） |
| `--rect-h MIN MAX` | 40 120 | 矩形高度（像素） |
| `--n-ellipse MIN MAX` | 3 6 | 椭圆光斑数量 |
| `--env-dir` | auto | 环境照片目录 |

## 环境照片

默认在以下位置自动查找：
- `/home/bobby/OCEC/env_reflection_textures/`
- `./env_reflection_textures/`
- `../env_reflection_textures/`

也可以用 `-e` 指定：
```bash
python nir_glass_reflection.py -i eye.png -o out.png -e /path/to/my_env_photos/
```

推荐使用**车内/仪表盘/窗外风景**等照片作为环境源，贴近 DMS 真实场景。

## 算法原理

1. **亮度归一化分离**：将眼睛图的亮度结构与内容分离
2. **环境替换**：用归一化后的环境亮度替换部分眼睛亮度（避免边界）
3. **高光叠加**：在归一化图上直接叠加程序化生成的光斑（无边界问题）
4. **Bloom 泛光**：整体轻微模糊融合

核心思路：**所有效果叠加在归一化后的图像上**，而不是与原图直接合成，从而彻底消除不同图像合成带来的边界问题。

## 上传 HuggingFace（可选）

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 登录
huggingface-cli login

# 上传整个目录
huggingface-cli upload your-repo-name /path/to/output_dir/ --repo-type dataset
```

## 许可

MIT License
