"""Build a local ImageNet-100 *proxy* from Tiny-ImageNet (64x64 images, upsampled to 224 by timm).

    python scripts/make_imagenet100_proxy.py [--data-dir data] [--per-class 50] [--seed 0]

Downloads http://cs231n.stanford.edu/tiny-imagenet-200.zip into <data-dir>/ (if missing), picks
100 of its 200 classes with a fixed seed (independent of any run seed), and writes an ImageFolder

    <data-dir>/imagenet100_proxy/{train,val}/<wnid>/<image>.JPEG   (symlinks)

with up to `per-class` seeded-random train images and up to `per-class` val images per class
(Tiny-ImageNet has 500 train and 50 val images per class). `val/` serves as the test set; the
engine carves its own validation split from `train/`. The proxy is for local validation only: its
images are sampled from ImageNet, so they may overlap the ViT pretraining set (audit F11/F12).
Rerunning is idempotent: correct links are kept, missing ones are created.
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def download_tiny_imagenet(data_dir: Path) -> Path:
    """Download and extract Tiny-ImageNet into data_dir (skipped when already extracted)."""
    root = data_dir / "tiny-imagenet-200"
    if (root / "wnids.txt").exists():
        return root
    archive = data_dir / "tiny-imagenet-200.zip"
    if not archive.exists():
        print(f"downloading {URL} -> {archive}")
        urllib.request.urlretrieve(URL, archive.with_suffix(".zip.part"))
        archive.with_suffix(".zip.part").rename(archive)
    print(f"extracting {archive}")
    with zipfile.ZipFile(archive) as zf:  # zipfile strips absolute paths and '..' components
        zf.extractall(data_dir)
    return root


def link(src: Path, dst: Path) -> None:
    """Symlink dst -> src; keep a correct existing link, replace a wrong one."""
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    elif dst.exists():
        raise FileExistsError(f"{dst} exists and is not a symlink")
    dst.symlink_to(src.resolve())


def build_proxy(data_dir: Path, per_class: int, seed: int) -> Path:
    """Write the proxy ImageFolder and return its root."""
    tiny = download_tiny_imagenet(data_dir)
    wnids = sorted((tiny / "wnids.txt").read_text().split())
    rng = np.random.RandomState(seed)
    classes = sorted(rng.choice(wnids, 100, replace=False).tolist())

    # val/images/*.JPEG is flat; val_annotations.txt maps file -> wnid.
    val_files: dict[str, list[str]] = {}
    for line in (tiny / "val" / "val_annotations.txt").read_text().splitlines():
        name, wnid = line.split("\t")[:2]
        val_files.setdefault(wnid, []).append(name)

    out = data_dir / "imagenet100_proxy"
    for wnid in classes:
        train_imgs = sorted((tiny / "train" / wnid / "images").glob("*.JPEG"))
        picks = rng.choice(len(train_imgs), min(per_class, len(train_imgs)), replace=False)
        splits = {"train": [train_imgs[i] for i in sorted(picks)],
                  "val": [tiny / "val" / "images" / n for n in sorted(val_files[wnid])[:per_class]]}
        for split, files in splits.items():
            (out / split / wnid).mkdir(parents=True, exist_ok=True)
            for f in files:
                link(f, out / split / wnid / f.name)
    (out / "PROXY.txt").write_text(f"Tiny-ImageNet proxy for ImageNet-100: seed {seed}, {per_class} train/val "
                                   f"images per class, 64x64 upsampled. Not the paper's ImageNet-100.\n"
                                   + "\n".join(classes) + "\n")
    for split in ("train", "val"):
        counts = [len(list((out / split / w).iterdir())) for w in classes]
        assert len(counts) == 100 and min(counts) > 0, f"{split}: empty class folder"
        print(f"{out / split}: {len(counts)} classes, {sum(counts)} images")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0, help="class/image selection seed (fixed, not a run seed)")
    args = parser.parse_args()
    build_proxy(args.data_dir, args.per_class, args.seed)
