import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")                       # save to file, no window needed
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

from dataset import IMAGE_SIZE, get_train_transform, scan_images


def pick_images(root, n, seed):
    """Pick n random images, half Fake and half Real."""
    samples = scan_images(root)
    rng = random.Random(seed)
    fakes = [s for s in samples if s[1] == "Fake"]
    reals = [s for s in samples if s[1] == "Real"]
    return rng.sample(fakes, n // 2) + rng.sample(reals, n - n // 2)


def draw_grid(rows, col_titles, out_path, title):
    """rows = list of (list_of_images, row_label). One image per column."""
    n_rows, n_cols = len(rows), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.4 * n_rows), squeeze=False)
    for r, (imgs, row_label) in enumerate(rows):
        for c, img in enumerate(imgs):
            axes[r][c].imshow(img)
            axes[r][c].axis("off")
            if r == 0:
                axes[r][c].set_title(col_titles[c], fontsize=9)
        axes[r][0].text(-0.05, 0.5, row_label, transform=axes[r][0].transAxes,
                        ha="right", va="center", fontsize=10, fontweight="bold")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4, help="number of example images")
    parser.add_argument("--path", default=None, help="dataset path (default: download with kagglehub)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="augmentation_samples")
    args = parser.parse_args()

    root = args.path
    if root is None:
        import kagglehub
        root = kagglehub.dataset_download("saurabhbagchi/deepfake-image-detection")
    os.makedirs(args.out, exist_ok=True)

    resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    images = []
    for path, label, _ in pick_images(root, args.n, args.seed):
        with Image.open(path) as im:
            images.append((resize(im.convert("RGB")), label))

    # --- Figure 1: each technique on its own (fixed values so they are easy to see) ---
    techniques = [
        ("Original", lambda im: im),
        ("Horizontal flip", TF.hflip),
        ("Vertical flip", TF.vflip),
        ("Rotate left 15°", lambda im: TF.rotate(im, 15)),      # positive angle = counter-clockwise
        ("Rotate right 15°", lambda im: TF.rotate(im, -15)),
        ("Zoom out 0.8x", lambda im: TF.affine(im, angle=0, translate=(0, 0), scale=0.8, shear=0)),
        ("Zoom in 1.2x", lambda im: TF.affine(im, angle=0, translate=(0, 0), scale=1.2, shear=0)),
    ]
    rows = [([fn(im) for _, fn in techniques], label) for im, label in images]
    draw_grid(rows, [name for name, _ in techniques], os.path.join(args.out, "1_each_technique.png"),
              "Each augmentation technique applied on its own")

    # --- Figure 2: the real train pipeline (without ToTensor + Normalize so we can still see the image) ---
    augment_only = transforms.Compose(get_train_transform().transforms[:-2])
    torch.manual_seed(args.seed)
    rows = [([im] + [augment_only(im) for _ in range(5)], label) for im, label in images]
    draw_grid(rows, ["Original"] + [f"Random #{i}" for i in range(1, 6)],
              os.path.join(args.out, "2_random_pipeline.png"),
              "Result of the Train augmentation pipeline (random each time)")


if __name__ == "__main__":
    main()