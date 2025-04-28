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
def cluster_segments(segs):
    """
    Segs is a list of (start_times, end_times) tuples for each semantic chunk/segment of a particular video.
    We cluster a sequence of segments sorted in ascending order of start times.

    If the current segments start time is too far from (> GAP_THRESHOLD) the last segment's end time of the current cluster,
    then we start a new cluster.

    Return: list of clusters [[(start, end), (start, end), ...],[(start, end), (start, end), ...], ...]
    """
    clusters = [] # initialize to be list of lists

    #sort segments by start times (the first element of each 2-tuple in the list)
    for start,end in sorted(segs):
        if not clusters or start - clusters[-1][-1][1] > GAP_THRESHOLD:
            # start new cluster if clusters is empty (for the 1st segs)
            # or if gap exceeds threshold between current segments start and the last segment's end in the latest cluster
            clusters.append([])
        # add current segment to the current(just-creating or pre-existing) cluster
        clusters[-1].append((start,end))
    return clusters

def pick_best(clusters):
    """
    Identifies the longest cluster (by number of segments) as the best cluster.

    Returns : the start time (the min of segment starts) and the end time (the max of segment ends)
    of the best cluster.

    This interval is used for obtaining the final clip of the video with longest best cluster. 
    """
    # identify longest cluster (with most segments)
    best_clust = max(clusters, key=len)

    # unpack the longest cluster and then zip all start times into 1 tuple and all end times into another tuple
    starts,ends = zip(*best_clust) # returns (start_1, start_2, ...) and (end_1, end_2, ...)
    return min(starts), max(ends)

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

    # 4) collect and cluster
    by_video = {}
    for h in hits:
        vid = h.payload["video_id"]
        # If the key for the current video_id is not present then insert that key with default list initialization
        # Either way an empty or prexisting list is returned by .setdefault() and then the (start_time, end_time) is appended
        by_video.setdefault(vid, []).append((h.payload["start"], h.payload["end"]))

    best_vid, best_span = None, None
    longest_seg_cluster = 0
    for vid, segs in by_video.items():
        clusters = cluster_segments(segs)
        for c in clusters:
            if len(c) > longest_seg_cluster:
                longest_seg_cluster = len(c)
                best_vid, best_span = vid, pick_best([c]) 

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
