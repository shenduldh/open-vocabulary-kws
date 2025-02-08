# Open Vocabulary Keyword Spotting

The project implements keyword spotting on arbitrary ASR models (theoretically). Specifically, I use beam search to boost search rewards for target terms to detect arbitrary keywords. Of course, this method can also be used for hotword recognition. The key codes therein are modified from [icefall](https://github.com/k2-fsa/icefall).

## Usage

In this repository, I use Paraformer from [FunASR](https://github.com/modelscope/FunASR) to demonstrate keyword spotting and hotword recognition.

1. Install dependences；

    ```bash
    pip install -r requirement.txt
    ```

2. Download Paraformer checkpoint from [paraformer-zh](https://huggingface.co/funasr/paraformer-zh)；

3. Run the simple example script；

    ```bash
    python paraformer_example.py
    ```

4. Or run the gradio demo to detect keywords from the microphone in real time.

    ```bash
    python gradio_demo.py
    ```
