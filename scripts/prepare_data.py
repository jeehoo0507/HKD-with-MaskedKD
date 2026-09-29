"""Run pinned COCO preparation using only the uv environment and Python stdlib."""
import argparse
import os
from pathlib import Path
import runpy
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"


def fetch_annotations(root):
    """Prepare the archive in advance so upstream never needs its curl fallback."""
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "annotations_trainval2017.zip"
    if archive.exists():
        return archive
    partial = archive.with_suffix(".zip.part")
    # A completed download may have been interrupted just before rename.
    if not partial.exists() or not zipfile.is_zipfile(partial):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(ANNOTATIONS, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urllib.request.urlopen(request, timeout=120) as response:
            resumed = response.status == 206 and offset > 0
            if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Annotation download resumed at the wrong byte; original partial file preserved")
            with partial.open("ab" if resumed else "wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
    with zipfile.ZipFile(partial) as zipped:
        if zipped.testzip() is not None:
            raise ValueError(f"Corrupt annotation archive: {partial}")
    os.replace(partial, archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    reference = ROOT / "external/benchmarks/datasets/coco_single/prepare.py"
    if not reference.is_file():
        parser.error("First run: bash scripts/prepare_reference.sh")
    data = ROOT / "data/coco_single"
    if not (data / "manifest.json").exists():
        fetch_annotations(data)
    # Reuse the recorded split/selection code without executing upstream setup.sh.
    reference_module = runpy.run_path(str(reference), run_name="hkd_reference_prepare")
    reference_module["prepare"](data, args.workers)


if __name__ == "__main__":
    main()
