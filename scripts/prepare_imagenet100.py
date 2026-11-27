"""Build ImageNet-100 as an ImageFolder of symlinks into an ILSVRC2012 (ImageNet-1k) root.

    python scripts/prepare_imagenet100.py --imagenet-dir /data/ILSVRC2012 --output-dir data/imagenet100
        [--val-labels LOC_val_solution.csv]

--imagenet-dir holds train/<wnid>/*.JPEG and val/. If val/ is already sorted into val/<wnid>/ the
class folders are linked; if it is the flat raw layout (val/ILSVRC2012_val_*.JPEG), pass
--val-labels with Kaggle's LOC_val_solution.csv (ImageId,PredictionString; the first token of
PredictionString is the wnid) and every image is linked into val/<wnid>/.

Class list: the fixed 100-class ImageNet-100 subset of Tian et al., "Contrastive Multiview Coding"
(ECCV 2020), the most widely used ImageNet-100 split. It is hard-coded (not drawn with a seed) so
every run and machine uses the same classes; every wnid is checked against ImageNet-1k.
Rerunning is idempotent: correct links are kept, wrong ones replaced, nothing is deleted (fixes
audit F10, where a rerun unlinked existing links and never recreated them).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

IMAGENET100_WNIDS = """
n02869837 n01749939 n02488291 n02107142 n13037406 n02091831 n04517823 n04589890 n03062245 n01773797
n01735189 n07831146 n07753275 n03085013 n04485082 n02105505 n01983481 n02788148 n03530642 n04435653
n02086910 n02859443 n13040303 n03594734 n02085620 n02099849 n01558993 n04493381 n02109047 n04111531
n02877765 n04429376 n02009229 n01978455 n02106550 n01820546 n01692333 n07714571 n02974003 n02114855
n03785016 n03764736 n03775546 n02087046 n07836838 n04099969 n04592741 n03891251 n02701002 n03379051
n02259212 n07715103 n03947888 n04026417 n02326432 n03637318 n01980166 n02113799 n02086240 n03903868
n02483362 n04127249 n02089973 n03017168 n02093428 n02804414 n02396427 n04418357 n02172182 n01729322
n02113978 n03787032 n02089867 n02119022 n03777754 n04238763 n02231487 n03032252 n02138441 n02104029
n03837869 n03494278 n04136333 n03794056 n03492542 n02018207 n04067472 n03930630 n03584829 n02123045
n04229816 n02100583 n03642806 n04336792 n03259280 n02116738 n02108089 n03424325 n01855672 n02090622
""".split()
assert len(set(IMAGENET100_WNIDS)) == 100


def link(src: Path, dst: Path) -> None:
    """Symlink dst -> src; keep a correct existing link, replace a wrong one, never delete data."""
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    elif dst.exists():
        raise FileExistsError(f"{dst} exists and is not a symlink")
    dst.symlink_to(src.resolve())


def prepare(imagenet_dir: Path, output_dir: Path, val_labels: Path | None) -> None:
    """Link the 100 classes' train and val images under output_dir/{train,val}/<wnid>."""
    for split in ("train", "val"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    for wnid in IMAGENET100_WNIDS:
        src = imagenet_dir / "train" / wnid
        if not src.is_dir():
            raise FileNotFoundError(f"missing {src}")
        link(src, output_dir / "train" / wnid)

    if val_labels is None:
        for wnid in IMAGENET100_WNIDS:
            src = imagenet_dir / "val" / wnid
            if not src.is_dir():
                raise FileNotFoundError(f"missing {src}: val/ is flat? pass --val-labels LOC_val_solution.csv")
            link(src, output_dir / "val" / wnid)
    else:
        wanted = set(IMAGENET100_WNIDS)
        with open(val_labels, newline="") as f:
            for row in csv.DictReader(f):
                wnid = row["PredictionString"].split()[0]
                if wnid in wanted:
                    (output_dir / "val" / wnid).mkdir(exist_ok=True)
                    name = row["ImageId"] + ".JPEG"
                    link(imagenet_dir / "val" / name, output_dir / "val" / wnid / name)

    # An empty class folder would silently evaluate to 0% (audit F10), so fail loudly instead.
    for split in ("train", "val"):
        counts = {w: sum(1 for _ in (output_dir / split / w).iterdir()) for w in IMAGENET100_WNIDS}
        empty = [w for w, n in counts.items() if n == 0]
        if empty:
            raise RuntimeError(f"{split}: empty class folders {empty}")
        print(f"{output_dir / split}: 100 classes, {sum(counts.values())} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--imagenet-dir", type=Path, required=True, help="ILSVRC2012 root with train/ and val/")
    parser.add_argument("--output-dir", type=Path, default=Path("data/imagenet100"))
    parser.add_argument("--val-labels", type=Path, help="LOC_val_solution.csv, for a flat val/ directory")
    args = parser.parse_args()
    prepare(args.imagenet_dir, args.output_dir, args.val_labels)
