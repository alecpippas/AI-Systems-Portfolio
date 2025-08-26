

## Customer Helper Functions for RAG Project Below:

from typing import List, Tuple, Iterable
import decord 
from PIL import Image, ImageChops
from llm_engineering.domain.types import Subtitle
from numpy.typing import NDArray



def is_duplicate(
        frame1,
        frame2,
        threshold: float = 5.0,
) -> bool:
    """
    Heuristic to detect visually near-identical frames: convert to grayscale, resize to 64x64,
    compute pixel-wise difference, and average. If the average difference is below a threshold,
    consider them duplicates to avoid redundancy of reference frames/chunks stored within MongoDB
    """
    # Convert to PIL for easy difference
    img1 = Image.fromarray(frame1.asnumpy()).convert("L").resize((64, 64))
    img2 = Image.fromarray(frame2.asnumpy()).convert("L").resize((64, 64))
    diff = ImageChops.difference(img1, img2)
    # average pixel difference
    stat = sum(diff.getdata()) / (64 * 64)
    return stat < threshold



def iterate_key_frames(
        video_reader: decord.VideoReader,
        segments: List[Subtitle],
) -> Iterable[Tuple[NDArray, Subtitle]]:
    """
    For each semantic chunk (Subtitle dataclass) extract a single reference frame -- the frame at the midpoint
    of the segment. This enforces semantic chunking by tying each frame to a meaningful piece of the lecture.

    THIS APPROACH ASSUMES THE CAPTIONS FROM THE HF DATASET, GROUPED INTO COHERENT SENTENCES, REFLECT SEMANTIC CHUNKS.
    """
    fps = video_reader.get_avg_fps() or 25.0
    for semantic_chunk in segments:
        mid_time = (semantic_chunk.start + semantic_chunk.end) / 2.0
        frame_index = int(mid_time * fps)
        if frame_index < len(video_reader):
            # obtain the reference frame (as a decord.VideoFrame ~(H, W, C) array of raw RBG pixel values)
            # correpsonding to the midpoint of the current semantic segment
            ref_frame = video_reader[frame_index]
            setattr(ref_frame, "ref_frame_timestamp", mid_time) #add a reference frame timestamp attribute to the re_framne class for the current semantic chunk
            yield ref_frame, semantic_chunk




