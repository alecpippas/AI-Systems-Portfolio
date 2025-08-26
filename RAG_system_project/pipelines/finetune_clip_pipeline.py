# Simple, one-file implementation of an end-to-end pipeline for fintuning the CLIP model
# Uses the reference frames for each semantic chunk (video segments from Intro to AI Course recordings)
# References frames are retrieved from a previously created local MongoDB instance


import os
import io
import torch
from dotenv import load_dotenv
from pymongo import MongoClient
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer
import torch.nn.functional as F
from llm_engineering.domain.types import VideoDataset

load_dotenv()  # load .env (environment variables) into os.environ

mongo_uri = os.environ.get(
    "DATABASE_HOST", "mongodb://127.0.0.1:27017")
client = MongoClient(mongo_uri) #establish connection to local MongoDB



class ContrastiveTrainer(Trainer):
    """
    Subclass the default trainer to override the compute_loss method so it utilizes labels that enable the summation across all
    image-text pairs with the computation of the contrastive loss function (InfoNCE)
    """
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **kwargs,                        
    ):
        """Override Trainer class's default compute_loss method so that it computes the """
        
        # pull out the fields we care about
        pixel_values  = inputs.pop("pixel_values")
        input_ids     = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")

        # forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits_per_image = outputs.logits_per_image
        logits_per_text  = outputs.logits_per_text

        # construct contrastive labels
        batch_size = logits_per_image.size(0)
        labels = torch.arange(batch_size, device=logits_per_image.device)

        # InfoNCE loss
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text,  labels)
        #average the the cross_entropy image loss and the textual loss
        loss = (loss_img + loss_txt) / 2

        return (loss, outputs) if return_outputs else loss


def main():
    """End-to-end finetuning workflow"""
    # 1) Retrieve the 'video_segments' collection from the local 'video-rag' MongoDB
    collection = client.video_rag.video_segments

    # 2) Fetch documents using MongoDB querty
    # {} : no filter - fetch every document
    # {"image_bytes": 1, "subtitle": 1} : project (return) only those fields (plus default _id)
    docs = list(collection.find({}, {"image_bytes": 1, "subtitle": 1})) 
    split_index = int(0.9 * len(docs)) # 90% training / 10% validation
    train_docs, val_docs = docs[:split_index], docs[split_index:]

    # 3) Initialize pre-trained CLIP Preprocessing Pipeline and CLIP Architecture (Text Encoder, Vision Transformer/ViT, Contrastive Objectivce)
    
    # CLIP tokenizer: converts raw text->list of tokens->tensor of token IDs
    # Image Transforms : resize, normalize, and formats to match ViT's expected input (3, 224, 224)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False) #download CLIP's tokenizer + image transforms
    
    # Text Encoder: transformer-based encoder that processes token IDs, outputting text's dense embedding
    # ViT (Vision-Image Transformer) : processes the image tensor through transoformer layers, outputting image's embedding
    # Contrastice Objective: used to align text and image embeddings in shared semantic/vector space;
    # similar image-text pairs brought together; mismatches pushed apart
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") #download the ViT + text encoders + Constractive Objective

    # 4) Wrap docs in Videotasets
    train_dataset = VideoDataset(train_docs, processor, return_text=True, return_meta=False) #
    val_dataset = VideoDataset(val_docs, processor, return_text=True, return_meta=False)

    # 5) Set up training
    training_args = TrainingArguments(
        output_dir="./clip_finetuned/checkpoints", #outputs checkpoints & final model
        per_device_train_batch_size=16, #per GPU/CPU
        per_device_eval_batch_size=16,
        num_train_epochs=7,
        eval_strategy="epoch", #evaluate at end of each epoch
        save_steps=20, #save checkpoint every n steps
        logging_steps=20,
    )

    # 6) Instantiate Trainer which ties together model, data, and training args
    # Trainer:
    #   -loops over training dataset, calls 'model(**batch), computes loss and backprops
    #   -run evaluation on eval_dataset each epoch
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
    )

    trainer.train() # start fine-tuning run
    trainer.save_model("clip_finetuned") # write out final model weights to ""

if __name__ == "__main__":
    main()
