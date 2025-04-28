from enum import StrEnum
from torch.utils.data import Dataset
from PIL import Image
import io

class DataCategory(StrEnum):
    PROMPT = "prompt"
    QUERIES = "queries"

    INSTRUCT_DATASET_SAMPLES = "instruct_dataset_samples"
    INSTRUCT_DATASET = "instruct_dataset"
    PREFERENCE_DATASET_SAMPLES = "preference_dataset_samples"
    PREFERENCE_DATASET = "preference_dataset"

    POSTS = "posts"
    ARTICLES = "articles"
    REPOSITORIES = "repositories"


### Custom Classes for RAG Project Defined Below

from dataclasses import dataclass


@dataclass
class Subtitle:
    """
    Dataclass that captures a semantic chunk of a video respresented by its start and end timestamps (in seconds) and the
    associated text.
    Used for aligning frames to meaningful segments in the ETL pipeline.
    """
    start: float #segment start time (secs)
    end: float # segment end time (secs)
    text: str # subtitle text for this segment



class VideoDataset(Dataset):
    """
    PyTorch-compatible dataset for pairing each frame with its subtitle.
    Stores all provided docs upon instantiation.
    Lazily pre-processes each doc (image-subtile pair) per invocation of __getitem___ within the Trainer's internal training loop.
    """
    def __init__(self, docs, processor, return_text: bool = True, return_meta: bool = False):
        self.docs = docs #list of MongoDB documents (formatted as dictionaries)
         #doc = {
         #       "video_id": id_ of full video from original HF dataset
         #       "start" : start time of semantic chunk
         #       "end" : end time of semantic chunk
         #       "ref_frame_timestamp": midpoint timestamp for the current semantic chunk
         #       "subtitle": captions corresponding to single semantic chunk
         #       "image_bytes": raw image bytes of reference frame
         #   }
        self.processor = processor # CLIPProcessor that turns (image, text) into token IDs and pixel tensors

        self.return_text = return_text #flag to return 1D tensor of token IDs of the subtitle (during finetuning pipeline)
        self.return_meta = return_meta #flag to return doc metadata when __getitem___ is called (during featurization pipeline)

    def __len__(self):
       return len(self.docs) # number of semantic chunks (samples) in the Dataset

    def __getitem__(self, idx):
        """Fetch sample semantic chunk with index: idx """
        doc = self.docs[idx]
         # wrap the raw image bytes using file-like buffer, io.BytesIO
        #   -prevents writing to temp .jpg requiring disk I/O 
        #   -enables compatability with PIL Image.open

        #retreive raw image bytes and textual subtitles from MongoDB doc
        image = Image.open(io.BytesIO(doc["image_bytes"])).convert("RGB") #force-cast to 3 8-bit color channels for CLIP compatability
        text = doc["subtitle"]
        
         #run CLIP preprocessing on both modalities simultaneously
        inputs = self.processor(
            text=[text], # processed into token IDs, wrap in list to indicate batch size=1
            images=[image], # processed into pixel tensors
            return_tensors="pt", # instructs the processor to return: "pt"=PyTorch Tensors, "tf"=TensorFLow, "np"=ndarrays
            padding="max_length", # pad each varing-length token sequence to CLIP's max supported length (77)
            truncation=True #cut off right-end of raw text with len > max_length
        )
        # inputs assigned dict:
        # {
        #   "input_ids": Tensor(shape=(1, max_length), dtyps=torch.long),   #token IDs
        #   "attention_mask": Tensor(shape=(1, max_length), dtype=torch.long),
        #   "pixel_values": Tensor(shape=(1, 3, H, W), dtype=torch.float),
        #   ...
        #}

        #squeez/remove first batch size dimension so easier to feed into CLIP model
        pixel_values = inputs["pixel_values"].squeeze(0) # assigns a 3D tensor (color_chans x H x W)
        result = {"pixel_values": pixel_values}

        # add token ID sequence to the result dict to pair with image embedding for finetuning
        if self.return_text:
            
            result["input_ids"] = inputs["input_ids"].squeeze(0)

        # add metadata to the result dict to pair with image embedding for featurization and upserting to Qdrant
        if self.return_meta:
            result.update({
                "id": str(doc["_id"]),
                "video_id": doc["video_id"],
                "start": doc["start"],
                "end": doc["end"],
                "subtitle": doc["subtitle"],
                "ref_frame_timestamp": doc["ref_frame_timestamp"]
            })
        return result

