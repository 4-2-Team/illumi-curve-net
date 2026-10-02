"""
Evaluation script for IllumiCurveNet.

Runs a snapshot (pretrained baseline or a fine-tuned checkpoint) over a
directory of images, and for each image logs:
  - The same unsupervised loss terms used during training (L_color, L_spa,
    L_exp, L_TV, L_contrast, L_texture) plus their weighted sum, computed on
    the model's own output the same way train.py computes them.
  - No-reference image quality metrics (NIQE, BRISQUE, PIQE - lower is
    better for all three) on both the original and the enhanced image, so
    the effect of enhancement on image quality can be judged without
    needing paired ground-truth.

Per-image results are written to a CSV, and a mean/std/min/max summary is
written to JSON. A handful of original/enhanced image pairs are also saved
for visual inspection.
"""

import os
import csv
import json
import time
import argparse

import numpy as np
import torch
import torchvision
from PIL import Image
from pyiqa import create_metric

import model
import utils.losses as losses
import utils.dataloader as dataloader

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# Same weights used in train.py's low_light_enhancement loss configuration,
# so the weighted total here is directly comparable to reported training loss.
LOSS_WEIGHTS = {
    "L_color": 5.0,
    "L_spa": 1.5,
    "L_exp": 10.0,
    "L_TV": 200.0,
    "L_contrast": 5.0,
    "L_texture": 3.0,
}


def collect_images(root):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def load_image(path):
    img = Image.open(path).convert('RGB')
    width = (img.size[0] // 8) * 8
    height = (img.size[1] // 8) * 8
    img = img.resize((width, height))
    arr = np.asarray(img) / 255.0
    tensor = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0)
    return tensor


def main():
    parser = argparse.ArgumentParser(description='IllumiCurveNet Evaluation Script')
    parser.add_argument('--data_root', type=str, default='Crescent-images-dataset/data_splits/test',
                         help='Directory of images to evaluate (searched recursively)')
    parser.add_argument('--pretrain_snapshot', type=str, default='snapshots/model-best.pth',
                         help='Model snapshot to evaluate')
    parser.add_argument('--output_dir', type=str, required=True,
                         help='Directory to write per-image CSV, summary JSON, and sample images')
    parser.add_argument('--sample_images', type=int, default=10,
                         help='Number of original/enhanced image pairs to save for inspection')
    parser.add_argument('--limit', type=int, default=0, help='0 = evaluate all images found')
    parser.add_argument('--skip_piqe', action='store_true', help='Skip PIQE (slowest of the three IQA metrics)')
    parser.add_argument('--synthetic_degrade', action='store_true',
                         help='Apply synthetic low-light degradation to inputs before enhancing (simulates deployment on genuinely dim frames)')
    parser.add_argument('--exp_mean_val', type=float, default=0.6, help='Target patch brightness for the masked exposure loss')
    parser.add_argument('--exp_content_frac', type=float, default=0.15,
                         help='Fraction of an image\'s brightest patch above which a patch counts as "content" for the exposure loss')
    config = parser.parse_args()

    os.makedirs(config.output_dir, exist_ok=True)
    sample_dir = os.path.join(config.output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = model.illumi_curve_net().cuda()
    net.load_state_dict(torch.load(config.pretrain_snapshot))
    net.eval()

    L_color = losses.L_color()
    L_spa = losses.L_spa()
    L_texture = losses.L_texture()
    L_exp = losses.L_exp_masked(patch_size=16, mean_val=config.exp_mean_val, content_frac=config.exp_content_frac)
    L_contrast = losses.L_contrast()
    L_TV = losses.L_TV()

    niqe = create_metric('niqe')
    brisque = create_metric('brisque')
    piqe = None if config.skip_piqe else create_metric('piqe')

    image_paths = collect_images(config.data_root)
    if config.limit:
        image_paths = image_paths[:config.limit]
    if not image_paths:
        print(f"No images found in {config.data_root}")
        raise SystemExit(1)

    print(f"Evaluating {len(image_paths)} image(s) from {config.data_root}")
    print(f"Snapshot: {config.pretrain_snapshot}")

    rows = []
    t_start = time.time()
    with torch.no_grad():
        for i, path in enumerate(image_paths):
            img = load_image(path).cuda()
            if config.synthetic_degrade:
                img = dataloader.synthetic_low_light(img.squeeze(0)).unsqueeze(0)
            enhanced, A = net(img)

            loss_vals = {
                'L_color': torch.mean(L_color(enhanced)).item(),
                'L_spa': torch.mean(L_spa(enhanced, img)).item(),
                'L_TV': torch.mean(L_TV(A)).item(),
                'L_texture': torch.mean(L_texture(img, enhanced)).item(),
                'L_exp': torch.mean(L_exp(enhanced, img)).item(),
                'L_contrast': torch.mean(L_contrast(enhanced)).item(),
            }
            weighted_total = sum(LOSS_WEIGHTS[k] * v for k, v in loss_vals.items())

            niqe_orig = niqe(img).item()
            niqe_enh = niqe(enhanced).item()
            brisque_orig = brisque(img).item()
            brisque_enh = brisque(enhanced).item()
            if piqe is not None:
                piqe_orig = piqe(img).item()
                piqe_enh = piqe(enhanced).item()
            else:
                piqe_orig = piqe_enh = float('nan')

            row = {
                'image': os.path.relpath(path, config.data_root),
                **{f'train_{k}': v for k, v in loss_vals.items()},
                'train_loss_weighted_total': weighted_total,
                'niqe_original': niqe_orig, 'niqe_enhanced': niqe_enh,
                'brisque_original': brisque_orig, 'brisque_enhanced': brisque_enh,
                'piqe_original': piqe_orig, 'piqe_enhanced': piqe_enh,
            }
            rows.append(row)

            if i < config.sample_images:
                torchvision.utils.save_image(img, os.path.join(sample_dir, f'{i:03d}_original.png'))
                torchvision.utils.save_image(enhanced, os.path.join(sample_dir, f'{i:03d}_enhanced.png'))

            if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
                print(f"  [{i + 1}/{len(image_paths)}] elapsed {time.time() - t_start:.1f}s")

    csv_path = os.path.join(config.output_dir, 'per_image_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    keys = [k for k in rows[0].keys() if k != 'image']
    summary = {}
    for k in keys:
        vals = [r[k] for r in rows if not (isinstance(r[k], float) and np.isnan(r[k]))]
        if vals:
            summary[k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
            }

    summary_path = os.path.join(config.output_dir, 'summary_metrics.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'data_root': config.data_root,
            'pretrain_snapshot': config.pretrain_snapshot,
            'num_images': len(rows),
            'elapsed_sec': time.time() - t_start,
            'metrics': summary,
        }, f, indent=2)

    print(f"\nDone. {len(rows)} image(s) evaluated in {time.time() - t_start:.1f}s")
    print(f"Per-image CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")
    print("\n=== Summary (mean) ===")
    for k, s in summary.items():
        print(f"  {k}: {s['mean']:.4f}")


if __name__ == '__main__':
    main()
