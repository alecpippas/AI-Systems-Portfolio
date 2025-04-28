#!/usr/bin/env python3
import os
import argparse
from datasets import load_dataset
from dotenv import load_dotenv

def download_mp4s(webdataset_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # stream the WebDataset
    ds = load_dataset(
        "webdataset",
        data_files={"train": webdataset_path},
        split="train",
        streaming=True,
    )

    seen_ids = set()
    for sample in ds:
        # each sample has keys "mp4" (bytes) and "json" (with video_id)
        vid = sample["json"]["video_id"]
        if vid in seen_ids:
            continue
        seen_ids.add(vid)

        mp4_bytes = sample["mp4"]
        out_path = os.path.join(output_dir, f"{vid}.mp4")
        with open(out_path, "wb") as f:
            f.write(mp4_bytes)
        print(f"✔ saved {out_path}")

    print(f"Downloaded {len(seen_ids)} videos to '{output_dir}'.")

if __name__ == "__main__":
    load_dotenv()  # optional: if you keep WEBDS_PATH in .env
    p = argparse.ArgumentParser(
        description="Download all .mp4 files from a WebDataset tar"
    )
    p.add_argument(
        "--webdataset",
        default=os.getenv("WEBDS_PATH", "data/youtube_dataset.tar"),
        help="Path or glob to your .tar WebDataset"
    )
    p.add_argument(
        "--out",
        default=os.getenv("VIDEO_DOWNLOAD_DIR", "downloaded_videos"),
        help="Directory to write .mp4 files into"
    )
    args = p.parse_args()
    download_mp4s(args.webdataset, args.out)
