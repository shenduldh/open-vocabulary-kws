import gradio as gr
from funasr import AutoModel
import json
import torch
import numpy as np

from paraformer_example import paraformer_forward
from search import KeywordSearcher


# set device and paths
device = "cuda:0"
model_path = "assets/paraformer-zh"  # download from https://huggingface.co/funasr/paraformer-zh
tokens2ids_path = "assets/tokens2ids.json"

# load model
pipeline = AutoModel(model=model_path, device=device, disable_update=True)
tokenizer = pipeline.kwargs["tokenizer"]

# set token ids to be jumped
jumped_ids = [
    tokenizer.token2id["<blank>"],
    tokenizer.token2id["<unk>"],
    tokenizer.token2id["<s>"],
    tokenizer.token2id["</s>"],
]

# load token mapping table
with open(tokens2ids_path, "r", encoding="utf-8") as f:
    token_table = json.load(f)

# save last searched answer
last_answer = None


def kws_fn(keywords: str, audio, stream=None):
    global last_answer

    if isinstance(audio, tuple):
        sr, y = audio
    else:
        y = audio

    if not isinstance(y, str):
        y = y.astype(np.float32)
        scale = 1.0 / float(1 << (16 - 1))
        y *= scale

        if stream is not None:
            y = np.concatenate([stream, y])
        stream = y

    ks = KeywordSearcher(
        keywords=[
            {"keyword": kw.strip(), "score": 0, "threshold": 0}
            for kw in keywords.split("\n")
        ],
        blank_id=tokenizer.token2id["<blank>"],
        jumped_ids=jumped_ids,
        tokens2ids=token_table,
        context_size=1,
    )

    decoder_out, decoder_out_lens = paraformer_forward(pipeline, [y], device)

    total_answers = [[] for _ in range(decoder_out.size(0))]
    states = []
    logits = decoder_out[:, 0, :]
    for i in range(decoder_out.size(1)):
        answers, states = ks.search_one_step(logits, states, i)
        _, sample_ids = ks.calc_sample_splits_ids(states)
        logits = torch.index_select(
            decoder_out[:, i, :], dim=0, index=sample_ids.to(device)
        )
        for j, ans in enumerate(answers):
            total_answers[j] += ans

    if stream is None:
        return total_answers[0]

    if len(total_answers[0]) > 0:
        last_answer = total_answers[0]
    return stream, last_answer


with gr.Blocks(title="KWS Demo") as demo:
    gr.Markdown("# Keyword Spotting")

    with gr.Row():
        with gr.Column():
            keywords = gr.Text(label="keywords", placeholder="one keyword per line")
            audio_uploaded = gr.Audio(sources="upload", type="filepath")
            button = gr.Button(value="detect keywords")
            audio_chunk = gr.Audio(sources=["microphone"], streaming=True)
            streaming_state = gr.State()
        with gr.Column():
            result = gr.Text()
            streaming_result = gr.Text()

    audio_chunk.stream(
        kws_fn,
        inputs=[keywords, audio_chunk, streaming_state],
        outputs=[streaming_state, streaming_result],
        time_limit=10,
        stream_every=0.5,
    )
    button.click(fn=kws_fn, inputs=[keywords, audio_uploaded], outputs=[result])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=12223)
