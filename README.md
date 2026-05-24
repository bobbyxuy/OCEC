# OCEC - Open Closed Eye Classifier

NIR eye image classification and glare data augmentation toolkit.

---

## Quick Start — Which files do you need?

### Task 1: Synthesize glare (overexposed / washed-out effect)

```bash
python make_comparison_grid.py        # visualize: left=clean, right=synthesized
```

Edit the three parameters in `make_comparison_grid.py` to tune the effect:

```python
blur_sigma = np.random.uniform(1.5, 3.5)   # blur amount (larger = more blurry)
factor     = np.random.uniform(0.65, 0.88) # contrast retention (1.0=none, 0.3=heavy)
lift       = np.random.uniform(8, 30)      # brightness lift in pixel values
```

Output saved to `output/glare_comparison.png`.

### Task 2: Synthesize glasses reflection (LED spot / streak)

```bash
python nir_glare_generator.py --input_dir ./eyes/ --output_dir ./eyes_glare/ --mode mixed
```

Modes: `led_spot`, `led_streak`, `ambient`, `mixed`

### Task 3: Validate synthesis quality against real glare

```bash
python test_flare_synthesis.py
```

Outputs stats table, histogram comparison, and t-SNE plot to `output/`.

---

## File Overview

### Core files (use these)

| File | Purpose |
|------|---------|
| `make_comparison_grid.py` | Visual comparison grid: clean vs synthesized haze/overexposure |
| `nir_glare_generator.py` | LED reflection synthesis (spot / streak / ambient) |
| `nir_eye_glare.py` | Gaussian glint + env texture blending |
| `test_flare_synthesis.py` | Validation: stats + histogram + t-SNE vs real glare |

### Reference data

| Directory | Contents |
|-----------|---------|
| `glass1_samples/` | 16 real NIR eye images (512×512) used as synthesis reference |
| `glare_demo/` | Synthesized examples for each mode (led_spot / led_streak / ambient / mixed) |
| `env_reflection_textures/` | Environment textures for `env_blend` mode |

### Other scripts (experimental / archived)

| File | Notes |
|------|-------|
| `nir_reflection_v2~v4_demo.py` | Progressive iterations of reflection compositing |
| `composite_v4.py` | Blender-render based compositing (requires Blender pass renders) |
| `procedural_reflection.py` | Procedural reflection without env textures |
| `eval_*.py` | Model evaluation scripts |
| `blender_*.py` | Blender-based render scripts |

---

## Synthesis Notes

The MRL Eye Dataset filename encodes glare level at position `[5]`:
- `flag=0`: clean, max pixel ~120, no overexposure
- `flag=1`: mild glare, occlusion effect (darker), 8% pixels < 30
- `flag=2`: heavy glare, mean +53, 7% pixels > 240

The haze/overexposure synthesis in `make_comparison_grid.py` targets `flag=2` characteristics.

---

## Eye Open/Closed Classifier

DINOv2-ViT-Large fine-tuned binary classifier (open/closed), F1 > 0.99 on MRL Eye Dataset.

```bash
python train_dinov2.py
```

Dataset: [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) — 84,898 IR eye images, 37 subjects.

## Dependencies

```
pip install pillow numpy opencv-python scikit-image scikit-learn matplotlib
```

## License

MIT
