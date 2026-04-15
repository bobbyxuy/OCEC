#!/usr/bin/env python3
"""
Eye Open/Closed Classifier - Inference Script
Model: DINOv2-ViT-Large fine-tuned on MRL Eye Dataset (NIR grayscale)
Accuracy: F1=0.9923 on MRL test set

Usage:
  # Single image
  python eye_open_closed_infer.py --image eye.jpg

  # Batch (directory)
  python eye_open_closed_infer.py --input_dir ./eyes/ --output_dir ./results/

  # With confidence threshold (skip uncertain predictions)
  python eye_open_closed_infer.py --input_dir ./eyes/ --output_dir ./results/ --conf_threshold 0.85

  # Use ONNX (faster, no PyTorch dependency)
  python eye_open_closed_infer.py --input_dir ./eyes/ --output_dir ./results/ --onnx dinov2_eye_classifier.onnx

  # Batch mode: generate labeled images with predictions
  python eye_open_closed_infer.py --input_dir ./eyes/ --output_dir ./labeled/ --draw_label

Output:
  For each input image, generates a copy with prediction text overlay:
    - "OPEN 0.98" (green) or "CLOSED 0.95" (red)
    - "UNCERTAIN 0.72" (yellow, below threshold)
  Also outputs a CSV summary: results.csv with columns: filename, prediction, confidence, label
"""

import os
import sys
import csv
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# PyTorch Backend
# ============================================================
def create_torch_model(weight_path, device="cpu"):
    import torch
    import torch.nn as nn
    import timm

    backbone = timm.create_model(
        "vit_large_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=0,
        img_size=64,
    )
    feat_dim = backbone.embed_dim  # 1024
    model = nn.Sequential(
        backbone,
        nn.LayerNorm(feat_dim),
        nn.Linear(feat_dim, 2),
    )
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device).eval()
    return model


def infer_torch(model, img_pil, device="cpu"):
    import torch
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    img = img_pil.convert("L").convert("RGB")  # grayscale -> 3ch
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)
    return int(pred.item()), float(conf.item())


# ============================================================
# ONNX Backend
# ============================================================
def infer_onnx(onnx_path, img_pil):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    # Preprocess: grayscale -> 3ch -> resize(64,64) -> normalize
    img = img_pil.convert("L").convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0
    # Resize to 64x64
    img_resized = np.array(Image.fromarray((img_array * 255).astype(np.uint8)).resize((64, 64)), dtype=np.float32) / 255.0
    # Normalize
    img_resized = (img_resized - 0.5) / 0.5
    # HWC -> CHW -> NCHW
    img_chw = img_resized.transpose(2, 0, 1)[np.newaxis]

    logits = sess.run(None, {inp_name: img_chw})[0]
    prob = np.exp(logits) / np.exp(logits).sum()
    pred = int(np.argmax(prob))
    conf = float(prob[0, pred])
    return pred, conf


# ============================================================
# Label Drawing
# ============================================================
LABEL_MAP = {0: "OPEN", 1: "CLOSED"}
COLOR_MAP = {
    "OPEN": (0, 200, 0),       # green
    "CLOSED": (200, 0, 0),     # red
    "UNCERTAIN": (200, 200, 0), # yellow
}


def draw_label(img_pil, prediction, confidence, threshold=0.85, true_label=None):
    """Draw prediction + ground truth label on image, return labeled PIL Image."""
    img = img_pil.copy().convert('RGB')
    draw = ImageDraw.Draw(img)

    w, h = img.size
    font_size = max(8, min(h // 10, 20))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Line 1: Prediction
    if confidence < threshold:
        pred_str = "UNSURE"
        pred_color = COLOR_MAP["UNCERTAIN"]
    else:
        pred_str = "OPEN" if prediction == 0 else "SHUT"
        pred_color = COLOR_MAP["OPEN" if prediction == 0 else "CLOSED"]
    line1 = "P:" + pred_str + " " + "%.4f" % confidence

    # Line 2: Ground truth (if provided)
    if true_label is not None:
        true_str = "OPEN" if true_label == 0 else "SHUT"
        match = (prediction == true_label) and (confidence >= threshold)
        true_color = (0, 255, 0) if match else (255, 80, 80)
        mark = "OK" if match else "WRONG"
        line2 = "T:" + true_str + " " + mark
    else:
        line2 = None
        true_color = None

    # Draw with background
    lines = [line1]
    if line2:
        lines.append(line2)
    line_h = font_size + 2
    max_w = max(draw.textbbox((0, 0), l, font=font)[2] for l in lines)
    box_h = len(lines) * line_h + 6

    draw.rectangle([0, 0, max_w + 8, box_h], fill=(0, 0, 0))
    draw.text((4, 2), line1, fill=pred_color, font=font)
    if line2:
        draw.text((4, 2 + line_h), line2, fill=true_color, font=font)

    return img


# ============================================================
# Main
# ============================================================
def process_single(image_path, infer_fn, draw=False, threshold=0.85, true_label=None):
    """Process a single image. Returns (prediction, confidence, labeled_image)."""
    img = Image.open(image_path)
    pred, conf = infer_fn(img)

    labeled = None
    if draw:
        labeled = draw_label(img, pred, conf, threshold, true_label=true_label)

    return pred, conf, labeled


def main():
    parser = argparse.ArgumentParser(description="Eye Open/Closed Classifier (DINOv2-ViT-Large)")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--input_dir", type=str, help="Input directory (batch mode)")
    parser.add_argument("--output_dir", type=str, help="Output directory for labeled images")
    parser.add_argument("--weight", type=str, default="best_dinov2_eye.pth", help="PyTorch weight path")
    parser.add_argument("--onnx", type=str, default=None, help="ONNX model path (alternative to --weight)")
    parser.add_argument("--conf_threshold", type=float, default=0.85, help="Confidence threshold for UNCERTAIN label (default: 0.85)")
    parser.add_argument("--draw_label", action="store_true", help="Draw prediction on output images")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--csv", type=str, default=None, help="CSV output path (default: <output_dir>/results.csv)")
    parser.add_argument("--true_label", type=int, default=None, choices=[0, 1], help="Ground truth: 0=open, 1=closed (for demo)")
    args = parser.parse_args()

    if not args.image and not args.input_dir:
        parser.error("Provide --image or --input_dir")

    # Initialize inference function
    if args.onnx:
        print(f"Using ONNX model: {args.onnx}")
        # Warm up
        infer_fn = lambda img: infer_onnx(args.onnx, img)
    else:
        print(f"Loading PyTorch model: {args.weight} (device={args.device})")
        model = create_torch_model(args.weight, device=args.device)
        infer_fn = lambda img: infer_torch(model, img, device=args.device)

    # Collect images
    if args.image:
        images = [(args.image, os.path.basename(args.image))]
    else:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        images = []
        for f in sorted(os.listdir(args.input_dir)):
            if os.path.splitext(f)[1].lower() in exts:
                images.append((os.path.join(args.input_dir, f), f))
        print(f"Found {len(images)} images")

    if not images:
        print("No images found!")
        return

    # Process
    results = []
    labeled_count = 0
    uncertain_count = 0

    for i, (path, name) in enumerate(images):
        pred, conf, labeled_img = process_single(path, infer_fn, draw=args.draw_label, threshold=args.conf_threshold, true_label=args.true_label)
        label = LABEL_MAP[pred] if conf >= args.conf_threshold else "UNCERTAIN"

        results.append({
            "filename": name,
            "prediction": label,
            "confidence": f"{conf:.4f}",
        })

        if label == "UNCERTAIN":
            uncertain_count += 1

        # Save labeled image
        if labeled_img and args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, name)
            labeled_img.save(out_path)
            labeled_count += 1

        if args.image:
            # Single image: print result
            print(f"  {name}: {label} (confidence={conf:.4f})")
        elif (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(images)}...")

    # Summary
    open_count = sum(1 for r in results if r["prediction"] == "OPEN")
    closed_count = sum(1 for r in results if r["prediction"] == "CLOSED")
    print(f"\n{'='*50}")
    print(f"Total: {len(results)}  OPEN: {open_count}  CLOSED: {closed_count}  UNCERTAIN: {uncertain_count}")
    print(f"Labeled images saved: {labeled_count}")

    # CSV
    csv_path = args.csv or (os.path.join(args.output_dir, "results.csv") if args.output_dir else "results.csv")
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "prediction", "confidence"])
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
