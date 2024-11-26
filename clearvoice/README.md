# clearvoice


## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Usage](#2-usage)

## 1. Introduction

`clearvoice` provides a unified inference platform for `speech enhancement`, `speech separation`, and `audio-visual target speaker extraction`. It aims to simplify the calling and adoptation of our pre-trained models for your project developments. We currently provide the following pre-trained models:

| Tasks (Sampling rate) | Models (HuggingFace Links)|
|-------|--------------------------|
|Speech Enhancement (16kHz & 48kHz)| `MossFormer2_SE_48K` ([link](https://huggingface.co/alibabasglab/MossFormer2_SE_48K)), `FRCRN_SE_16K` ([link](https://huggingface.co/alibabasglab/FRCRN_SE_16K)), `MossFormerGAN_SE_16K` ([link](https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K)) | 
|Speech Separation (16kHz)|`MossFormer2_SS_16K` ([link](https://huggingface.co/alibabasglab/MossFormer2_SS_16K))|
|Audio-Visual Target Speaker Extraction (16kHz)|`AV_MossFormer2_TSE_16K` ([link](https://huggingface.co/alibabasglab/AV_MossFormer2_TSE_16K))|

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

Sample example 1: use speech enhancement model `MossFormer2_SE_48K` to process one wave file of `samples/input.wav` and save the output wave file to `samples/output_MossFormer2_SE_48K.wav`

```python
from clearvoice import ClearVoice

myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

output_wav = myClearVoice(input_path='samples/input.wav', online_write=False)

myClearVoice.write(output_wav, output_path='samples/output_MossFormer2_SE_48K.wav')
```

Sample example 2: use speech enhancement model `MossFormer2_SE_48K` to process all input wave files in `samples/path_to_input_wavs/` and write all output files to `samples/path_to_output_wavs`

```python
from clearvoice import ClearVoice

myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

myClearVoice(input_path='samples/path_to_input_wavs', online_write=True, output_path='samples/path_to_output_wavs')
```

Sample example 3: use speech enhancement model `MossFormer2_SE_48K` to process wave files listed in `samples/audio_samples.scp' file, and write outputs to 'samples/path_to_output_wavs_scp/'

```python
from clearvoice import ClearVoice

myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

myClearVoice(input_path='samples/scp/audio_samples.scp', online_write=True, output_path='samples/path_to_output_wavs_scp')
```

