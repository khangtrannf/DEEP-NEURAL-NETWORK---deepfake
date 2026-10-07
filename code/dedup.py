import hashlib
import io
import json
import os

import numpy as np
from PIL import Image


def dhash(img, size=8):
    """64-bit 'difference hash': similar-looking images (even resized / re-encoded) give similar hashes."""
    gray = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(gray, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def fingerprint(path):
    """Return (md5, dhash, aspect_ratio) of an image file, or None if the file cannot be read."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        with Image.open(io.BytesIO(data)) as im:
            aspect = im.width / im.height
            im.draft("L", (128, 128))       # JPEG only: decode a small version -> much faster for big photos
            return hashlib.md5(data).hexdigest(), dhash(im), aspect
    except Exception:
        return None


def load_fingerprints(samples, root, cache_name=".dedup_cache.json", verbose=True):
    """Fingerprints for every sample. Cached in <root>/.dedup_cache.json, so only the first run is slow."""
    cache_path = os.path.join(root, cache_name)
    try:
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        cache = {}

    fps, changed = [], False
    if verbose and any(os.path.relpath(p, root) not in cache for p, _, _ in samples):
        print(f"[dedup] Fingerprinting {len(samples)} images (first run only, result is cached)...")
    for path, _, _ in samples:
        rel, size = os.path.relpath(path, root), os.path.getsize(path)
        entry = cache.get(rel)
        if entry is None or entry["size"] != size:
            fp = fingerprint(path)
            entry = {"size": size, "fp": None if fp is None else [fp[0], str(fp[1]), fp[2]]}
            cache[rel] = entry
            changed = True
        fp = entry["fp"]
        fps.append(None if fp is None else (fp[0], int(fp[1]), fp[2]))

    if changed:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f)
        except OSError:
            pass                              # read-only folder: just skip caching
    return fps


def build_groups(dhashes, aspects, hash_threshold=4, aspect_tol=0.02):
    """
    Group near-duplicates. Two images are linked if their dhash differs in <= hash_threshold of 64 bits
    AND their aspect ratios match (a resized copy keeps its aspect ratio; this removes most false matches).
    Returns an array with one group id per image (images in the same group share the id).
    """
    n = len(dhashes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if abs(aspects[i] - aspects[j]) < aspect_tol and bin(dhashes[i] ^ dhashes[j]).count("1") <= hash_threshold:
                parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])


def deduplicate(samples, root, hash_threshold=4, aspect_tol=0.02, verbose=True):
    """
    samples: list of (path, label, split) from scan_images().
    Returns (clean_samples, groups): groups[k] is the near-duplicate group id of clean_samples[k].
    """
    fps = load_fingerprints(samples, root, verbose=verbose)

    readable = [i for i, fp in enumerate(fps) if fp is not None]          # 1. unreadable files
    seen, keep = set(), []
    for i in readable:                                                    # 2. exact duplicates
        if fps[i][0] not in seen:
            seen.add(fps[i][0])
            keep.append(i)

    groups = build_groups([fps[i][1] for i in keep], [fps[i][2] for i in keep],   # 3. near-duplicates
                          hash_threshold, aspect_tol)
    labels = np.array([samples[i][1] for i in keep])
    conflict = {g for g in np.unique(groups) if len(set(labels[groups == g])) > 1}   # 4. conflicting labels
    final = [k for k, g in enumerate(groups) if g not in conflict]

    if verbose:
        sizes = np.bincount(np.unique(groups[final], return_inverse=True)[1])    # groups that are KEPT
        print(f"[dedup] Unreadable files dropped        : {len(samples) - len(readable)}")
        print(f"[dedup] Exact duplicates dropped        : {len(readable) - len(keep)}")
        print(f"[dedup] Near-duplicate groups (kept, same split): {int((sizes > 1).sum())} "
              f"({int(sizes[sizes > 1].sum())} images)")
        print(f"[dedup] Conflicting-label groups dropped: {len(conflict)} "
              f"({len(keep) - len(final)} images)")
        print(f"[dedup] Images left: {len(final)} / {len(samples)}")
    return [samples[keep[k]] for k in final], groups[final]