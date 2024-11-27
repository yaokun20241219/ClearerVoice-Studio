# ClearVoice

## 👉🏻[HuggingFace Space Demo](https://huggingface.co/spaces/alibabasglab/ClearVoice)👈🏻

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Usage](#2-usage)

## 1. Introduction

ClearVoice offers a unified inference platform for `speech enhancement`, `speech separation`, and `audio-visual target speaker extraction`. It is designed to simplify the adoption of our pre-trained models for your speech processing purpose or the integration into your projects. Currently, we provide the following pre-trained models:

| Tasks (Sampling rate) | Models (HuggingFace Links)|
|-------|--------------------------|
|Speech Enhancement (16kHz & 48kHz)| `MossFormer2_SE_48K` ([link](https://huggingface.co/alibabasglab/MossFormer2_SE_48K)), `FRCRN_SE_16K` ([link](https://huggingface.co/alibabasglab/FRCRN_SE_16K)), `MossFormerGAN_SE_16K` ([link](https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K)) | 
|Speech Separation (16kHz)|`MossFormer2_SS_16K` ([link](https://huggingface.co/alibabasglab/MossFormer2_SS_16K))|
|Audio-Visual Target Speaker Extraction (16kHz)|`AV_MossFormer2_TSE_16K` ([link](https://huggingface.co/alibabasglab/AV_MossFormer2_TSE_16K))|

No need to manually download the pre-trained models. They are automatically downloaded during inference.
## 2. Usage

### Step-by-Step Guide

1. **Clone the Repository**

``` sh
git clone https://github.com/modelscope/ClearerVoice-Studio.git
```

2. **Create Conda Environment**

``` sh
cd ClearerVoice-Studio
conda create -n clearvoice python=3.8
conda activate clearvoice
pip install -r requirements.txt
```

3. **Run Demo**

``` sh
cd clearvoice
python demo.py
```

or 

``` sh
cd clearvoice
python demo_with_more_comments.py
```

You may activate each demo case by setting to True in `demo.py` and `demo_with_more_comments.py`.

4. **Use Scripts**

Use `MossFormer2_SE_48K` model for fullband (48kHz) speech enhancement task:

```python
from clearvoice import ClearVoice

myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

#process single wave file
output_wav = myClearVoice(input_path='samples/input.wav', online_write=False)
myClearVoice.write(output_wav, output_path='samples/output_MossFormer2_SE_48K.wav')

#process wave directory
myClearVoice(input_path='samples/path_to_input_wavs', online_write=True, output_path='samples/path_to_output_wavs')

#process wave list file
myClearVoice(input_path='samples/scp/audio_samples.scp', online_write=True, output_path='samples/path_to_output_wavs_scp')
```

Parameter Description:
- `task`: Choose one of the three tasks `speech_enhancement`, `speech_separation`, and `target_speaker_extraction`
- `model_names`: List of model names, choose one or more models for the task
- `input_path`: Path to the input audio/video file, input audio/video directory, or a list file (.scp) 
- `online_write`: Set to `True` to enable saving the enhanced/separated audio/video directly to local files during processing, otherwise, the enhanced/separated audio is returned. (Only supports `False` for `speech_enhancement`, `speech_separation` when processing single wave file`)
- `output_path`: Path to a file or a directory to save the enhanced/separated audio/video file
