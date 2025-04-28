# Simple pipeline to extract image embeddings from the fine-tuned CLIP Model and upsert them
# into a local Qdrant DB.

import os
import io
from dotenv import load_dotenv
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from llm_engineering.domain.types import VideoDataset

load_dotenv()  # load your .env

# 1) CONFIG & CLIENTS

# Mongo
MONGO_URI       = os.getenv("DATABASE_HOST", "mongodb://127.0.0.1:27017")
MONGO_DB        = os.getenv("DATABASE_NAME", "video_rag")
MONGO_COLL      = "video_segments"

# Qdrant
QDRANT_HOST     = os.getenv("QDRANT_DATABASE_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_DATABASE_PORT",  "6333"))
QDRANT_COLL     = "video_segments_index"
VECTOR_SIZE     = 512               # CLIP ViT-B/32 outputs 512-dim vectors

# Fine-tuned CLIP
CLIP_MODEL_DIR  = os.getenv("CLIP_FINETUNED_MODEL_PATH", "/home/alecpippas/dev_projects/AI_Course_Projects/LLM-Engineers-Handbook/clip_finetuned")  
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Connect to MongoDB and Qdrant (must be actively running in docker containers if local)
mongo_client = MongoClient(MONGO_URI)
mongo_coll = mongo_client[MONGO_DB][MONGO_COLL]

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
qdrant.recreate_collection(
    collection_name=QDRANT_COLL,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)

# Load model + processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
model     = CLIPModel.from_pretrained(CLIP_MODEL_DIR).to(DEVICE).eval()



def collate_fn(batch):
    """
    Stacks pixel_values (image embeddings) across batch into one long PyTorch tensor.
    Creates a metadata dictionary with the same metadata keys and a list of all the corresponding values across the batch.

    Inputs:
        - batch: List[Dicts]  (see VideoDataset class definition)

    Returns:
        - pixel_values: concatenated batch-wide image tensor
        - meta: batch-wide metadata dictionary
    """
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0) # stack = take Python list if same-dim tensors and concatenate into one large PyTorch tensor
    meta = {
        "id":        [b["id"]        for b in batch], # list of MongoDB IDs from samples in the batch
        "video_id":  [b["video_id"]  for b in batch], # list of original Video IDs from samples in the batch
        "start":  [b["start"]  for b in batch], 
        "end":    [b["end"]    for b in batch],
        "ref_frame_timestamp": [b["ref_frame_timestamp"] for b in batch],
        "subtitle":  [b["subtitle"]  for b in batch],
    }
    return pixel_values, meta


# 3) FEATURIZE & UPLOAD

def main(batch_size: int = 32):
    # fetch required fields from MongoDB that will be upserted into Qdrant DB
    docs = list(mongo_coll.find(
        {}, 
        {"image_bytes": 1, "video_id": 1, "start" : 1, "end" : 1, "ref_frame_timestamp": 1, "subtitle": 1}
    ))
    dataset = VideoDataset(docs, processor, return_text=False, return_meta=True)
    loader  = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    #initialize unsigned integer counter for Qdrant IDs
    pointID_counter = 0
    # iterate over each batch of pixel byte vectors and metadata dicts
    for pixels, meta in loader:
        pixels = pixels.to(DEVICE) #move raw pixel bytes to GPU/CPU
        with torch.no_grad():
            embeddings = model.get_image_features(pixels)  # (B, 512)
        embeddings = embeddings.cpu().numpy()

        #intialize list of PointStruct's
        points = []
        for vec, mongo_id, vid, start, end, ts, sub in zip(
            embeddings,
            meta["id"],
            meta["video_id"],
            meta["start"],
            meta["end"],
            meta["ref_frame_timestamp"],
            meta["subtitle"]
        ):
            payload = {
                "mongo_id": mongo_id,
                "video_id": vid,
                "start": start,
                "end": end,
                "ref_frame_timestamp": ts,
                "subtitle": sub,
            }
            points.append(
                PointStruct(id=pointID_counter, vector=vec.tolist(), payload=payload)
            )
            pointID_counter += 1

        qdrant.upsert(collection_name=QDRANT_COLL, points=points)

    print(" Featurization complete — all image embeddings are in Qdrant.")


if __name__ == "__main__":
    main()
