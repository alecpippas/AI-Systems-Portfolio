#!/usr/bin/env python
"""
tools/inspect_webdataset.py

Stream a few examples from a WebDataset and print:
 - dataset.features schema
 - the keys in each sample
 - the Python type of each field
 - a small preview of its value

Usage:
  rye run python tools/inspect_webdataset.py \
      --data-files data/youtube_dataset.tar \
      --split data \
      --num-samples 5
"""
import argparse
import json
from datasets import load_dataset

def inspect_webdataset(data_files, split="train", num_samples=3):
    # 1) load in streaming mode
    ds = load_dataset("webdataset", data_files={"data": data_files}, split=split, streaming=True)

    # 2) print the features (if available)
    try:
        print("=== Dataset Features Schema ===")
        print(json.dumps(ds.features, default=lambda o: repr(o), indent=2))
    except Exception:
        print("No .features available for streaming dataset.")

    # 3) iterate and inspect first num_samples
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        print(f"\n--- Sample #{i} keys & types ---")
        for k, v in sample.items():
            vtype = type(v)
            # For nested dicts, show keys only
            preview = v
            if isinstance(v, (bytes, bytearray)):
                preview = f"<{len(v)} bytes>"
            elif isinstance(v, dict):
                preview = {kk: type(vv).__name__ for kk, vv in v.items()}
            elif hasattr(v, "to_pil") or hasattr(v, "to_image"):
                preview = f"<{vtype.__name__} object>"
            # Avoid huge prints
            if isinstance(preview, str) and len(preview) > 200:
                preview = preview[:200] + "…(truncated)"
            print(f"{k!r:15}: {vtype.__name__:20} = {preview!r}")
    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-files", type=str, required=True,
                        help="Path or glob to the tar(s) for your WebDataset")
    parser.add_argument("--split", type=str, default="train", help="Which split to load")
    parser.add_argument("--num-samples", type=int, default=3, help="How many samples to peek at")
    args = parser.parse_args()

    inspect_webdataset(args.data_files, split=args.split, num_samples=args.num_samples)
