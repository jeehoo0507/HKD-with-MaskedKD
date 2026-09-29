"""Use the benchmark manifest without resplitting, subsampling, or test tuning."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Dataset

CLASSES = ("giraffe", "airplane", "clock", "zebra", "train", "bird", "elephant", "toilet", "stop sign", "bear")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != 1 or tuple(manifest.get("classes", ())) != CLASSES
            or manifest.get("split_seed") != 20260922
            or manifest.get("selection") != "one_annotated_instance_per_image"):
        raise ValueError("Expected the benchmark's documented COCO single manifest.")
    rows = manifest["images"]
    counts = Counter()
    ids = set()
    for row in rows:
        label, split = row["label"], row["split"]
        if type(label) is not int or not 0 <= label < 10 or split not in ("train", "val", "test"):
            raise ValueError("Invalid label or split")
        if row["id"] in ids or row.get("instances") != 1:
            raise ValueError("Duplicate sample ID or invalid instance count")
        ids.add(row["id"])
        expected_source = "val2017" if split == "test" else "train2017"
        if row.get("source_split") != expected_source:
            raise ValueError("Source split does not match the benchmark")
        path = (root / row["image"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing image or path outside data root: {row['image']}")
        counts[split, label] += 1
    if any(counts["train", i] != 321 or counts["val", i] != 53 or counts["test", i] < 1 for i in range(10)):
        raise ValueError("Expected 321 train and 53 validation samples per class, with all test classes present.")
    if sum(counts["test", i] for i in range(10)) != 297:
        raise ValueError("Expected the recorded 297-image test set")
    return rows


class CocoSingle(Dataset):
    def __init__(self, root, rows, split, seed=0, epoch=0):
        self.root, self.split, self.seed, self.epoch = Path(root), split, seed, epoch
        self.rows = [row for row in rows if row["split"] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as image:
            image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        key = f"{self.seed}:{self.epoch}:{row['id']}".encode()
        if self.split == "train" and hashlib.sha256(key).digest()[0] < 128:
            image = TF.hflip(image)
        x = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return x, row["label"], row["id"]


def loader(root, rows, split, batch, seed=0, epoch=0, workers=0):
    dataset = CocoSingle(root, rows, split, seed, epoch)
    return DataLoader(dataset, batch_size=batch, shuffle=split == "train",
                      drop_last=split == "train", num_workers=workers, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed + epoch))
