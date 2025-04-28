# Definition of ETL pipeline (using ZenML or similar)
import torch.multiprocessing as tmp
tmp.set_sharing_strategy('file_descriptor') #fix WSL deadlock issue since WSL doesn't genuine shared memory page system


import os
import io
import logging
import tempfile
from datasets import load_dataset
import pymongo
import decord
from PIL import Image
from dotenv import load_dotenv
import datetime


from llm_engineering.application.dataset.utils import (
    iterate_key_frames,
    is_duplicate,
)
from llm_engineering.domain.types import Subtitle



logger = logging.getLogger (__name__)
logging.basicConfig(level=logging.INFO)



def timestamp_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS.sss timestamp string to float seconds."""
    t = datetime.datetime.strptime(ts, "%H:%M:%S.%f")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def run_video_etl(
        webdataset_path: str,
        mongo_uri: str,
        db_name: str = "video_rag", #MongoDB name
        coll_name: str = "video_segments",
        frame_interval: float = 1.0, #seconds between key frames (we won't store every frame for efficiency)
):
    """Stream WebDataset, extract frame + subtitle pairs, and store them in a local MongoDB instance."""
    # 1) Connect to Mongo
    client = pymongo.MongoClient(mongo_uri)
    coll = client[db_name][coll_name]
    logger.info(f"Connected to MongoDB {mongo_uri}, DB={db_name}, Coll={coll_name}")

    #2) Stream the HuggingFace WebDataset
    ds = load_dataset(
        "webdataset",
        data_files={"train": webdataset_path},
        split="train",
        streaming=True, #dataset will be loaded as an iterable generator function in which frames and metadata are downloaded as they are utilized in code
        token=os.environ["HUGGINGFACE_ACCESS_TOKEN"]
    )

    last_frame = None
    count = 0

    for sample in ds: #each iteration yields one video-level record
        #sample is a dict like: {
        #   'mp4 : the raw byrees of the full video file,
        #   'info.json' : a metadata dict about that video,
        #   'en.vtt' : the WebVTT subtitle track as bytes,
        #   'json' : a dict containing the caption list [{'start':..., 'end':..., 'text':...}], video_id and title,
        #   '__key__' : internal key HuggingFace uses to address the sample
        #   '__url__' : URLs HuggingFace uses to address the sample



        # 3) Identify the video ID from JSON metadata
        video_id = sample['json']['video_id']

        # 4) Get the subtitles/captions from the JSON captions list
        captions_list = sample["json"]["captions"]
        # turn captions into custom Subtitle dataclass
        segments = [
            Subtitle(
                start=timestamp_to_seconds(cap['start']),
                end=timestamp_to_seconds(cap['end']),
                text=cap['text']
            )
            for cap in captions_list
        ]

        # 5) Stream the mp4 bytes from the current sample into a VideoReader via a temp file
        mp4_bytes = sample["mp4"]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(mp4_bytes)
            tmp.flush()
            #lazily load and decode video frame from temp file
            video_reader = decord.VideoReader(tmp.name) #decord.VideoReader object indexes all keyframes/frames; normal Pythonic indexing compatible
        
        os.remove(tmp.name)

        # 6) Iterate over key frames
        # iterate_key_frames() returns only one reference frame (the midpoint frame) for each semantic chunk
        for ref_frame, semantic_chunk in iterate_key_frames(video_reader, segments):
            
            # filter out duplicate mid-point refernce frames
            if last_frame is not None and is_duplicate(ref_frame, last_frame):
                continue

            # 7) Encode current frame (mp4 bytes) as JPEG bytes
            img = Image.fromarray(ref_frame.asnumpy()) #convert: decord.VideoFrame -> ndarry -> PIL image object
            buf = io.BytesIO() #initialize in-memory binary buffer
            img.save(buf, format="JPEG") #save image (a decord.VideoFrame object with raw pixels) into the buffer as JPEG-compressed bytes (not saving it to disk)
            img_bytes = buf.getvalue() # extract raw bytes from buf

            # 8) Insert into MongoDB
            doc = {
                "video_id": video_id,
                "start" : semantic_chunk.start,
                "end" : semantic_chunk.end,
                "ref_frame_timestamp": ref_frame.ref_frame_timestamp, #midpoint timestamp for the current semantic chunk
                "subtitle": semantic_chunk.text,
                "image_bytes": img_bytes,
            }
            coll.insert_one(doc)
            last_frame = ref_frame #last_frame references the frame just previously stored
            count += 1

            if count % 100 == 0:
                #log every 100 segments inserted into MongoDB
                logger.info(f"Inserted {count} segments so far...")

    logger.info(f"ETL complete: inserted {count} video segments.")


if __name__ == "__main__":
    load_dotenv()  # load environment variables from .env into os.environ

    run_video_etl(
        webdataset_path="https://huggingface.co/datasets/aegean-ai/ai-lectures-spring-24/resolve/main/youtube_dataset.tar",
        mongo_uri=os.environ["DATABASE_HOST"],
        db_name="video_rag",
        coll_name="video_segments",
)