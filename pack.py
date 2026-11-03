"""Build the two zips to upload to Google Drive / Kaggle for GPU runs.

    python pack.py              -> dist/lifemate_code.zip, dist/zaic_data_noaudio.zip (~10 MB)
    python pack.py --audio      -> the data zip also holds audio (~1 GB, needed for ASR later)
    python pack.py --audio-zip public_test train:val_seen   -> dist/audio_<...>.zip with only those samples' audio
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from lifemate.data import DEFAULT_DATA_ROOT

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--audio-zip", nargs="*", default=None, help="specs <split> or <split>:<key of splits/train_split.json>")
    args = ap.parse_args()
    dist = HERE / "dist"
    dist.mkdir(exist_ok=True)
    if args.audio_zip:
        return pack_audio(Path(args.root), args.audio_zip, dist)

    with zipfile.ZipFile(dist / "lifemate_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in list((HERE / "lifemate").glob("*.py")) + list((HERE / "tests").glob("*.py")) + [HERE / "splits/train_split.json"]:
            z.write(p, p.relative_to(HERE).as_posix())

    root = Path(args.root)
    n = 0
    with zipfile.ZipFile(dist / "zaic_data_noaudio.zip" if not args.audio else dist / "zaic_data.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for split in ("train", "public_test"):
            for p in (root / split).rglob("*"):
                rel = p.relative_to(root)
                if p.is_dir() or "__pycache__" in rel.parts or p.name == ".DS_Store":
                    continue
                if not args.audio and rel.parts[1] == "audio":
                    continue
                z.write(p, rel.as_posix())
                n += 1
    print(f"code + {n} data files -> {dist}")


def pack_audio(root: Path, specs: list[str], dist: Path):
    """Audio folders of the selected samples only (wav is already compressed poorly, so store without deflate)."""
    import json

    name = "audio_" + "_".join(s.replace(":", "-") for s in specs) + ".zip"
    n = 0
    with zipfile.ZipFile(dist / name, "w", zipfile.ZIP_STORED) as z:
        for spec in specs:
            split, _, key = spec.partition(":")
            ids = set(json.loads((HERE / "splits/train_split.json").read_text(encoding="utf-8"))[key]) if key else None
            for d in sorted((root / split / "audio").iterdir()):
                if ids is not None and d.name not in ids:
                    continue
                for p in d.glob("*.wav"):
                    z.write(p, p.relative_to(root).as_posix())
                    n += 1
    print(f"{n} wav files -> {dist / name} ({(dist / name).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
