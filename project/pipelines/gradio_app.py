
import gradio as gr
from pipelines.video_retrieval_stream import retrieve_and_package_clip
import tempfile


def gradio_retrieve(query):
    # collect all the chunks into one bytes blob
    clip_bytes = retrieve_and_package_clip(query)

    # write out to a temp .mp4 file

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(clip_bytes)
    tmp.flush()

    # return the filename for gr.Video
    return tmp.name

with gr.Blocks() as demo:
    gr.Markdown("## Video Retrieval Demo")
    inp = gr.Textbox(label="Ask a question…")
    vid = gr.Video(label="Matched Clip")
    inp.submit(fn=gradio_retrieve, inputs=inp, outputs=vid)

if __name__ == "__main__":
    demo.launch()
