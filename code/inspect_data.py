import argparse
import csv
import hashlib
import io
import os
import random
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dataset import scan_images


def dhash(img, size=8):
    """64-bit 'difference hash': similar-looking images give similar hashes (even if resized/re-encoded)."""
    gray = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(gray, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def analyze(samples):
    rows, corrupt = [], []
    for path, label, split in samples:
        try:
            with open(path, "rb") as f:
                data = f.read()
            with Image.open(io.BytesIO(data)) as im:
                im.load()                                   # forces full decode -> catches truncated files
                rows.append(dict(path=path, label=label, split=split or "none",
                                 width=im.width, height=im.height, mode=im.mode, format=im.format,
                                 kb=round(len(data) / 1024, 1),
                                 md5=hashlib.md5(data).hexdigest(), dhash=dhash(im)))
        except Exception as e:
            corrupt.append((path, repr(e)))
    return rows, corrupt


def title(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, help="dataset path (default: download with kagglehub)")
    parser.add_argument("--out", default="inspect_output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.path
    if root is None:
        import kagglehub
        root = kagglehub.dataset_download("saurabhbagchi/deepfake-image-detection")
    os.makedirs(args.out, exist_ok=True)

    samples = scan_images(root)
    rows, corrupt = analyze(samples)
    hints = []

    # ---- 1. counts ----
    title("1. Image counts (original split x label)")
    for (split, label), n in sorted(Counter((r["split"], r["label"]) for r in rows).items()):
        print(f"  {split:6s} {label:5s} {n:5d}")

    # ---- 2. corrupt files ----
    title("2. Corrupt / unreadable files")
    print(f"  {len(corrupt)} file(s)")
    for path, err in corrupt[:10]:
        print("  ", path, "->", err)
    if corrupt:
        hints.append(f"{len(corrupt)} corrupt file(s): remove them (cleaning IS needed).")

    # ---- 3. per-class properties ----
    title("3. Properties per class")
    stats = {}
    for label in sorted({r["label"] for r in rows}):
        rs = [r for r in rows if r["label"] == label]
        w, h = np.array([r["width"] for r in rs]), np.array([r["height"] for r in rs])
        non_square = float(np.mean(w != h))
        stats[label] = dict(w=np.median(w), h=np.median(h), kb=np.median([r["kb"] for r in rs]),
                            formats=set(r["format"] for r in rs), non_square=non_square)
        print(f"\n  [{label}] {len(rs)} images")
        print(f"    modes   : {dict(Counter(r['mode'] for r in rs))}")
        print(f"    formats : {dict(Counter(r['format'] for r in rs))}")
        print(f"    width   : min {w.min()} | median {np.median(w):.0f} | max {w.max()}")
        print(f"    height  : min {h.min()} | median {np.median(h):.0f} | max {h.max()}")
        print(f"    non-square images: {non_square:.0%}")
        print(f"    top sizes: {Counter(zip(w.tolist(), h.tolist())).most_common(3)}")
        print(f"    median file size: {stats[label]['kb']:.1f} KB")

    if any(r["mode"] != "RGB" for r in rows):
        hints.append("Some images are not RGB (grayscale/RGBA/palette): already handled by .convert('RGB') in the dataset.")
    if any(s["non_square"] > 0.1 for s in stats.values()):
        hints.append("Many non-square images: Resize((224,224)) squashes them. Consider Resize(256)+CenterCrop(224) or padding.")
    if min(min(r["width"], r["height"]) for r in rows) < 64:
        hints.append("Some images are very small (<64 px on the short side): check them visually.")
    if len(stats) == 2:
        a, b = list(stats.values())
        size_ratio = max(a["w"], b["w"]) / max(1, min(a["w"], b["w"]))
        kb_ratio = max(a["kb"], b["kb"]) / max(0.1, min(a["kb"], b["kb"]))
        if size_ratio > 1.5 or kb_ratio > 2 or a["formats"] != b["formats"]:
            hints.append("Real and Fake differ in size / file size / format: the model might learn these "
                         "file properties instead of deepfake artifacts (shortcut). Consider re-encoding all images "
                         "the same way before training.")

    # ---- 4. exact duplicates ----
    title("4. Exact duplicates (same file bytes)")
    groups = defaultdict(list)
    for r in rows:
        groups[r["md5"]].append(r)
    dups = [g for g in groups.values() if len(g) > 1]
    cross = [g for g in dups if len({r["split"] for r in g}) > 1]
    conflict = [g for g in dups if len({r["label"] for r in g}) > 1]
    print(f"  duplicate groups: {len(dups)} | across different splits: {len(cross)} | with conflicting labels: {len(conflict)}")
    for g in dups[:5]:
        print("   -", [os.path.relpath(r["path"], root) for r in g])
    if dups:
        hints.append(f"{len(dups)} exact-duplicate group(s): remove duplicates BEFORE splitting, otherwise the same "
                     f"image can land in both Train and Test (data leakage).")
    if conflict:
        hints.append(f"{len(conflict)} duplicate group(s) have different labels (Real vs Fake): label noise, check them.")

    # ---- 5. near duplicates ----
    title("5. Near-duplicates (very similar images, Hamming distance <= 4 of 64 bits)")
    pairs = []
    if len(rows) <= 5000:
        hashes = [r["dhash"] for r in rows]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i]["md5"] != rows[j]["md5"] and bin(hashes[i] ^ hashes[j]).count("1") <= 4:
                    pairs.append((i, j))
        cross_pairs = [p for p in pairs if rows[p[0]]["split"] != rows[p[1]]["split"]]
        print(f"  pairs: {len(pairs)} | across different splits: {len(cross_pairs)}")
        print("  (this check is rough - look at some pairs yourself before trusting it)")
        for i, j in pairs[:10]:
            print("   -", os.path.relpath(rows[i]["path"], root), "<->", os.path.relpath(rows[j]["path"], root))
        with open(os.path.join(args.out, "near_duplicate_pairs.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["image_a", "image_b", "split_a", "split_b", "label_a", "label_b"])
            for i, j in pairs:
                wr.writerow([rows[i]["path"], rows[j]["path"], rows[i]["split"], rows[j]["split"],
                             rows[i]["label"], rows[j]["label"]])
        if pairs:
            hints.append(f"{len(pairs)} near-duplicate pair(s) ({len(cross_pairs)} across splits): open "
                         f"near_duplicate_pairs.csv and look at a few. If they are real duplicates, drop one of each pair.")
    else:
        print("  skipped (more than 5000 images)")

    # ---- save csv + sample figure ----
    with open(os.path.join(args.out, "images.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    rng = random.Random(args.seed)
    labels = sorted({r["label"] for r in rows})
    n_show = 8
    fig, axes = plt.subplots(len(labels), n_show, figsize=(2 * n_show, 2.2 * len(labels)), squeeze=False)
    for ri, label in enumerate(labels):
        pick = rng.sample([r for r in rows if r["label"] == label], n_show)
        for ci, r in enumerate(pick):
            with Image.open(r["path"]) as im:
                axes[ri][ci].imshow(im.convert("RGB"))
            axes[ri][ci].axis("off")
            axes[ri][ci].set_title(f"{label} {r['width']}x{r['height']}", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "samples.png"), dpi=130)
    plt.close(fig)

    # ---- summary ----
    title("SUMMARY: what to do")
    if hints:
        for h in hints:
            print(" -", h)
    else:
        print(" - No problems detected: no extra cleaning needed beyond Resize + Normalize.")
    print(f"\nFiles saved in: {args.out}/  (images.csv, near_duplicate_pairs.csv, samples.png)")
    print("Please send the text above (and samples.png) to whoever is reviewing the preprocessing.")


if __name__ == "__main__":
    main()
