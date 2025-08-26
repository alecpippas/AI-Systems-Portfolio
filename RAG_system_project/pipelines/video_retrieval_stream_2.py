# Pipeline for query → answer (may be simple function instead)
# pipelines/video_retrieval_stream.py

import os
import subprocess
import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient

# ── CONFIG ─────────────────────────────────────────────────────────────────────
QDRANT_HOST       = os.getenv("QDRANT_DATABASE_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_DATABASE_PORT",  "6333"))
QDRANT_COLLECTION = "video_segments_index"

# point this at your fine-tuned CLIP model folder
CLIP_MODEL_DIR    = os.getenv("CLIP_FINETUNED_MODEL_PATH", "clip_finetuned")
# folder where your raw videos live, named {video_id}.mp4
VIDEOS_DIR        = os.getenv("VIDEOS_DIR", "./videos")

# how many neighbors, how far apart is “too far”
# average caption segment length from the captions .json across Youtube video dataset: 1.6616 secs
TOP_K             = 150
GAP_THRESHOLD     = 8 # seconds

# ── HELPERS ────────────────────────────────────────────────────────────────────
def cluster_segments(segs, gap_threshold=GAP_THRESHOLD):
    """
    segs: list of (start, end, score) tuples, sorted by start time.
    If the gap between one segment's start and the last segment's end
    in the current cluster exceeds gap_threshold, start a new cluster.
    Returns: list of clusters, each itself a list of those same 3-tuples.
    """
    clusters = []
    for start, end, score in sorted(segs, key=lambda t: t[0]):
        # new cluster?
        if not clusters or start - clusters[-1][-1][1] > gap_threshold:
            clusters.append([])
        clusters[-1].append((start, end, score))
    return clusters

def score_of(cluster):
    """Sum up all the score values in the cluster triples."""
    return sum(t[2] for t in cluster)

def pick_best_cluster(cluster):
    """
    Given one cluster (list of (start,end,score) tuples),
    returns (start_min, end_max, total_score_of_cluster).
    """
    starts = [t[0] for t in cluster]
    ends   = [t[1] for t in cluster]
    return min(starts), max(ends), score_of(cluster)

# ── MAIN STREAMING RETRIEVAL ────────────────────────────────────────────────────
def retrieve_and_package_clip(query: str):
    """
    Returns an MP4 byte blob of the best matching clip for `query`.
    """
    # 1) load fine-tuned CLIP
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model     = CLIPModel.from_pretrained(CLIP_MODEL_DIR).eval()

    # 2) encode text
    toks      = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    text_emb  = model.get_text_features(**toks).detach().cpu().numpy()[0]

    # 3) search Qdrant
    qc        = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    hits      = qc.search(
                  collection_name=QDRANT_COLLECTION,
                  query_vector=text_emb.tolist(),
                  limit=TOP_K,
                  with_payload=True
              )

   # collect as plain tuples
    by_video = {}
    for h in hits:
        vid = h.payload["video_id"]
        seg_tuple = (h.payload["start"], h.payload["end"], h.score)
        by_video.setdefault(vid, []).append(seg_tuple)
        # If the key for the current video_id is not present then insert that key with default list initialization
        # Either way an empty or prexisting list is returned by .setdefault() and then the (start_time, end_time) is appended

    # 5) find the highest‐scoring cluster across all videos
    best_vid, best_span, best_score = None, (0,0), 0.0
    for vid, segs in by_video.items():
        for cluster in cluster_segments(segs):
            start, end, total_score = pick_best_cluster(cluster)
            if total_score > best_score:
                best_score = total_score
                best_vid   = vid
                best_span  = (start, end)

    if not best_vid:
        raise ValueError("No segments found for that query.")

    start, end = best_span
    in_file = os.path.join(VIDEOS_DIR, f"{best_vid}.mp4")

    # 5) ffmpeg‐pipe out just that segment, chunked
    cmd = [
      "ffmpeg", "-y",
      "-i", in_file,
      "-ss", str(start),
      "-to", str(end),
      "-f", "mp4",
      "-movflags", "frag_keyframe+empty_moov",
      "pipe:1"
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # 6) read everything into memory and return as one bytes blob
    clip_bytes, _ = proc.communicate()
    
    return clip_bytes
