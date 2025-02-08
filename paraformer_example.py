from funasr import AutoModel
from funasr.auto.auto_model import prepare_data_iterator
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
import torch
from typing import Union


def paraformer_forward(
    pipeline: AutoModel, audio_data: list, device: Union[torch.device, str]
):
    frontend = pipeline.kwargs["frontend"]
    tokenizer = pipeline.kwargs["tokenizer"]
    audio_fs = pipeline.kwargs.get("fs", 16000)
    data_type = pipeline.kwargs.get("data_type", "sound")
    ##########
    key_list, data_list = prepare_data_iterator(audio_data)
    audio_sample_list = load_audio_text_image_video(
        data_list,
        fs=frontend.fs,
        audio_fs=audio_fs,
        data_type=data_type,
        tokenizer=tokenizer,
    )
    speech, speech_lengths = extract_fbank(
        audio_sample_list, data_type=data_type, frontend=frontend
    )
    speech = speech.to(device=device)
    speech_lengths = speech_lengths.to(device=device)
    ##########
    encoder_out, encoder_out_lens = pipeline.model.encode(speech, speech_lengths)
    predictor_outs = pipeline.model.calc_predictor(encoder_out, encoder_out_lens)
    pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = predictor_outs[0:4]
    pre_token_length = pre_token_length.round().long()
    decoder_outs = pipeline.model.cal_decoder_with_predictor(
        encoder_out, encoder_out_lens, pre_acoustic_embeds, pre_token_length
    )
    decoder_out, decoder_out_lens = decoder_outs[0], decoder_outs[1]
    return decoder_out, decoder_out_lens


if __name__ == "__main__":
    import librosa
    import json
    from search import BeamSearcher, KeywordSearcher

    # set device and paths
    device = "cuda:0"
    model_path = "assets/paraformer-zh"  # download from https://huggingface.co/funasr/paraformer-zh
    tokens2ids_path = "assets/tokens2ids.json"
    test_audio1 = "samples/test1.mp3"
    test_audio2 = "samples/test2.mp3"

    # load model
    pipeline = AutoModel(model=model_path, device=device, disable_update=True)

    # load samples
    test_audio1, sr = librosa.load(test_audio1, sr=16000, mono=True)

    # get predicted logits
    decoder_out, decoder_out_lens = paraformer_forward(
        pipeline, [test_audio1, test_audio2], device
    )

    # get token mapping table
    with open(tokens2ids_path, "r", encoding="utf-8") as f:
        tokens2ids = json.load(f)

    ids2tokens = {}
    for k, v in tokens2ids.items():
        ids2tokens[v] = k

    ################
    ## ASR example
    ################

    # build searcher
    bs = BeamSearcher(
        hotwords=[{"keyword": "哪吒", "score": 4}],
        blank_id=tokens2ids["<blank>"],
        jumped_ids=[
            tokens2ids["<blank>"],
            tokens2ids["<unk>"],
            tokens2ids["<s>"],
            tokens2ids["</s>"],
        ],
        temperature=1,
        token_table=tokens2ids,
        context_size=1,
    )

    # search answers
    states = []
    logits = decoder_out[:, 0, :]
    for i in range(1, decoder_out.size(1)):
        states = bs.search_one_step(logits, states, i)
        _, sample_ids = bs.calc_sample_splits_ids(states)
        logits = torch.index_select(
            decoder_out[:, i, :], dim=0, index=sample_ids.to(device)
        )
    answers = bs.get_best_answers(states, ids2tokens)

    # because several sequences are being inferred at the same time
    # there is padding in the short sequences
    # the results for the short sequences will output meaningless content such as repeats
    # in practice, the padding needs to be removed
    print(answers)

    ################
    ## KWS example
    ################

    # build searcher
    ks = KeywordSearcher(
        keywords=[
            {"keyword": "哪吒", "score": 0, "threshold": 0},
            {"keyword": "大山", "score": 0, "threshold": 0},
        ],
        blank_id=tokens2ids["<blank>"],
        jumped_ids=[
            tokens2ids["<blank>"],
            tokens2ids["<unk>"],
            tokens2ids["<s>"],
            tokens2ids["</s>"],
        ],
        tokens2ids=tokens2ids,
        context_size=1,
    )

    # search answers
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
    answers = ks.final_search(states)
    for j, ans in enumerate(answers):
        total_answers[j] += ans

    # because several sequences are being inferred at the same time
    # there is padding in the short sequences
    # the results for the short sequences will output meaningless content such as repeats
    # in practice, the padding needs to be removed
    print(total_answers)
