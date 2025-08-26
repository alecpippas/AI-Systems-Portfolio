

#### _Demonstration shown in demonstartion.ipynb_

May have to download to a local device in order to view, given file is greater than Github 50MB limit for remote files.

#### __QDrant and MongoDB Containers__

Use the following command in the root of the project directory to launch the Qdrant and MongoDB
Docker Containers:
    
      "docker-compose up -d"

#### __Requirements__:

1) A local copy of all original .mp4 videos within the local file system. Path to video directory should be specified using the VIDEOS_DIR environment variable within .env/.
2) All dependencies listed within requirements.lock.
3) A local Docker container containting a Qdrant DB with a collection of the semantic chunk reference frame image embeddings obtained from a locally fine-tuned, pre-trained CLIP model.
4) The fine-tuning output files from HuggingFace's Trainer. These files include the fine-tuned parameters (weights + biases) and the hyperparameter/config files. My local repo has the fine-tuned paramaters stored under the top-level 'clip_fintuned' directory. These files include:
    + config.json
    + merges.txt
    + special_tokens_map.json
    + tokenizer_config.json
    + training_args.bin
    + vocab.json
5) The local rye virtual environment Python interpreter needs to be set up as a Jupyter Notebook kernel. The following command can be used to utilize said kernel ('rye show' can be used to find the environemnt_namne under the project field output):
    + "rye run python -m ipykernel install --user --name=<environment_name> --display-name "Python (<environment_name>)"


#### __Pipelines__

To run the various pipelines use the following commands in the root of the project directory:
  + ETL Pipeline: "rye run python -m pipelines.video_etl_pipeline"
  + Fine-tuning Pipeline: "rye run python -m pipelines.fintune_clip_pipeline"
  + Featurization Pipeline: "rye run python -m pipelines.video_featurize_pipeline"


#### __Gradio Application__ ####

To run the Gradio Application open the local loopback address in a web browser that the following command returns:
  + "rye run python -m front_end_app.gradio_app"
