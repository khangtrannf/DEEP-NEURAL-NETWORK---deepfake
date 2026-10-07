import os
import warnings
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dedup import deduplicate

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMAGE_SIZE = 224
# ImageNet mean/std: required for transfer learning (pretrained ConvNeXt/EfficientNet),
# and used for every part so that the comparison between parts is fair.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# 1. Transforms
# ----------------------------------------------------------------------------
def get_train_transform(image_size=IMAGE_SIZE):
    """Augmentation for the Train set (geometric changes only, no color changes)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),                 # horizontal flip
        transforms.RandomVerticalFlip(p=0.5),                   # vertical flip
        transforms.RandomAffine(degrees=15, scale=(0.8, 1.2)),  # rotate left/right up to 15 deg, zoom out 0.8x ~ zoom in 1.2x
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transform(image_size=IMAGE_SIZE):
    """Val/Test: NO augmentation, only resize + normalize."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ----------------------------------------------------------------------------
# 2. Scan images and read labels from the folder structure
# ----------------------------------------------------------------------------
def scan_images(root):
    """
    Walk through `root` and return a list of (image_path, label, original_split).
      - label          : 'Real' or 'Fake', taken from the parent folder name (real / fake)
      - original_split : 'train' / 'test' / 'val' if the path has such a folder, else None
      - folders starting with 'Sample' are skipped
    Actual structure of this Kaggle dataset:
        <root>/train-...-001/train/{real,fake}/*.jpg
        <root>/test-...-001/test/{real,fake}/*.jpg
        <root>/Sample_fake_images/Sample_fake_images/fake/*.jpg   <- skipped
    """
    samples, skipped_sample, skipped_no_label = [], 0, 0
    for dirpath, _, filenames in os.walk(root):
        # folder names below root, lowercase, e.g. ['train-2025...-001', 'train', 'real']
        folders = [f.lower() for f in os.path.relpath(dirpath, root).split(os.sep)]

        if any(f.startswith("sample") for f in folders):
            skipped_sample += sum(fn.lower().endswith(IMG_EXTS) for fn in filenames)
            continue

        label = next((f.capitalize() for f in reversed(folders) if f in ("real", "fake")), None)
        split = next((s for f in folders for s, names in (("train", ("train",)), ("test", ("test",)),
                      ("val", ("val", "valid", "validation"))) if f in names), None)

        for fn in sorted(filenames):
            if not fn.lower().endswith(IMG_EXTS):
                continue
            if label is None:
                skipped_no_label += 1
                continue
            samples.append((os.path.join(dirpath, fn), label, split))

    if not samples:
        raise RuntimeError(
            f"No images found inside folders named 'real'/'fake' under: {root}\n"
            f"Open this folder and check its structure, then adjust scan_images()."
        )
    if skipped_sample:
        print(f"[dataset] Skipped {skipped_sample} images in Sample_* folders")
    if skipped_no_label:
        print(f"[dataset] Skipped {skipped_no_label} images that are not inside a real/fake folder")
    samples.sort(key=lambda x: x[0])        # sorted so results are reproducible
    return samples


# ----------------------------------------------------------------------------
# 3. Train / Val / Test split
# ----------------------------------------------------------------------------
def stratified_split(labels, val_ratio=0.1, test_ratio=0.1, seed=42, groups=None):
    """
    Return 3 index arrays (train, val, test). Each class is split with the same ratios.
    groups: optional group id per image. Images with the same id always go to the same split
            (used to keep near-duplicate images together).
    """
    labels = np.asarray(labels)
    groups = np.arange(len(labels)) if groups is None else np.asarray(groups)
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(labels):
        in_class = np.where(labels == c)[0]
        n_val = int(round(len(in_class) * val_ratio))
        n_test = int(round(len(in_class) * test_ratio))
        class_groups = np.unique(groups[in_class])
        rng.shuffle(class_groups)
        n_in_val = n_in_test = 0
        for g in class_groups:
            members = in_class[groups[in_class] == g].tolist()
            if n_in_val < n_val:
                val_idx += members
                n_in_val += len(members)
            elif n_in_test < n_test:
                test_idx += members
                n_in_test += len(members)
            else:
                train_idx += members
    return np.array(sorted(train_idx)), np.array(sorted(val_idx)), np.array(sorted(test_idx))


def make_splits(splits, targets, groups=None, mode="pooled", val_ratio=0.1, test_ratio=0.1,
                official_val_ratio=0.2, seed=42):
    """
    mode="pooled"   : merge Kaggle's train+test, re-split stratified 80/10/10 (default).
    mode="official" : keep Kaggle's test set as-is; val = `official_val_ratio` of train (stratified).
                      Train images that are near-duplicates of a test image are dropped (no leakage).
    groups: near-duplicate group id per image (from dedup.py) or None.
    """
    targets = np.asarray(targets)
    if mode == "pooled":
        return stratified_split(targets, val_ratio, test_ratio, seed, groups)
    if mode == "official":
        splits = np.asarray(splits, dtype=object)
        groups = np.arange(len(targets)) if groups is None else np.asarray(groups)
        test_idx = np.where(splits == "test")[0]
        pool = np.where(splits != "test")[0]            # everything that is not test = train pool
        if len(test_idx) == 0 or len(pool) == 0:
            raise RuntimeError("No train/test folders found -> use split_mode='pooled'.")
        leaked = np.isin(groups[pool], groups[test_idx])
        if leaked.any():
            print(f"[dataset] official mode: dropped {int(leaked.sum())} train images that are "
                  f"near-duplicates of test images")
        pool = pool[~leaked]
        tr, va, _ = stratified_split(targets[pool], val_ratio=official_val_ratio, test_ratio=0.0,
                                     seed=seed, groups=groups[pool])
        return pool[tr], pool[va], test_idx
    raise ValueError("split_mode must be 'pooled' or 'official'")


class DeepfakeDataset(Dataset):
    """Simple dataset: list of image paths + integer labels + its own transform."""

    def __init__(self, paths, targets, transform=None):
        self.paths = list(paths)
        self.targets = list(targets)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[i]


# ----------------------------------------------------------------------------
# 4. Main function used by the other team members
# ----------------------------------------------------------------------------
def get_dataloaders(batch_size=32, seed=42, num_workers=2, image_size=IMAGE_SIZE,
                    dataset_path=None, val_ratio=0.1, test_ratio=0.1, split_mode="pooled",
                    official_val_ratio=0.2, remove_duplicates=True, verbose=True):
    """
    Returns: train_loader, val_loader, test_loader, classes
      - classes: class names ordered by label, e.g. ['Fake', 'Real'] (Fake=0, Real=1)
      - Only train_loader has augmentation. Val/Test are fixed (shuffle=False).
      - split_mode="pooled"  : merge Kaggle's train+test, re-split 80/10/10 (default).
        split_mode="official": use Kaggle's test set as-is, take 20% of train as validation.
      - remove_duplicates=True: drop exact duplicates / conflicting-label images and keep
        near-duplicate images in the same split (see dedup.py). Needs ~1 min on the first run, then cached.
      - dataset_path: pass a local path if the dataset is already downloaded,
        otherwise it is downloaded with kagglehub.
    """
    if dataset_path is None:
        import kagglehub
        dataset_path = kagglehub.dataset_download("saurabhbagchi/deepfake-image-detection")

    samples = scan_images(dataset_path)
    groups = None
    if remove_duplicates:
        samples, groups = deduplicate(samples, dataset_path, verbose=verbose)
    classes = sorted({label for _, label, _ in samples})             # ['Fake', 'Real']
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths = [p for p, _, _ in samples]
    targets = [class_to_idx[label] for _, label, _ in samples]
    original_split = [s for _, _, s in samples]

    train_idx, val_idx, test_idx = make_splits(original_split, targets, groups, split_mode, val_ratio,
                                               test_ratio, official_val_ratio, seed)

    def build(idx, transform):
        return DeepfakeDataset([paths[i] for i in idx], [targets[i] for i in idx], transform)

    # 3 independent datasets, each with its own transform -> augmentation only in train
    train_ds = build(train_idx, get_train_transform(image_size))
    val_ds = build(val_idx, get_eval_transform(image_size))
    test_ds = build(test_idx, get_eval_transform(image_size))

    if verbose:
        print(f"[dataset] Root: {dataset_path}")
        print(f"[dataset] Classes: {class_to_idx} | split_mode = {split_mode}")
        for name, ds in (("Train", train_ds), ("Val", val_ds), ("Test", test_ds)):
            counts = np.bincount(ds.targets, minlength=len(classes))
            print(f"[dataset] {name:5s}: {len(ds):5d} images | " +
                  " ".join(f"{c}={n}" for c, n in zip(classes, counts)))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader, classes


if __name__ == "__main__":
    print("--- Checking the dataset pipeline ---")
    train_loader, val_loader, test_loader, classes = get_dataloaders(batch_size=32)
    print(f"-> Classes: {classes}")
    print(f"-> Number of batches: Train {len(train_loader)} | Val {len(val_loader)} | Test {len(test_loader)}")
    images, labels = next(iter(train_loader))
    print(f"-> Image batch shape: {tuple(images.shape)} (B, C, H, W)")
    print(f"-> Label batch shape: {tuple(labels.shape)}")
    print("-> Train has augmentation:", "RandomAffine" in str(train_loader.dataset.transform))
    print("-> Val has augmentation:  ", "RandomAffine" in str(val_loader.dataset.transform))
    print("OK - pipeline works.")