[![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=ClearerVoice-Studio&text2=%20AI%20Powered%20Speech%20Front-End%20Processing%20Toolkit&width=1000&height=210)](https://github.com/Akshay090/svg-banners)
    
<strong>ClearerVoice-Studio</strong> is an open-source toolkit for **[speech enhancement](https://github.com/modelscope/ClearerVoice-Studio/tree/main/train/speech_enhancement)**, **[speech separation](https://github.com/modelscope/ClearerVoice-Studio/tree/main/train/speech_separation)**, and <a href="https://github.com/modelscope/ClearerVoice-Studio/blob/main/train/target_speaker_extraction">**target speaker extraction**<a/>, for both researchers and developers in speech processing. It contains state-of-the-art pre-trained models, training and inference scripts. The repo is organized into three main components: **[ClearVoice](https://github.com/modelscope/ClearerVoice-Studio/tree/main/clearvoice)**, **[SpeechScore](https://github.com/modelscope/ClearerVoice-Studio/tree/main/speechscore)**, and **[Train](https://github.com/modelscope/ClearerVoice-Studio/tree/main/train)**.

#### 👉🏻[ClearVoice Demo](https://huggingface.co/spaces/alibabasglab/ClearVoice)👈🏻  
#### 👉🏻[SpeechScore Demo](https://huggingface.co/spaces/alibabasglab/SpeechScore)👈🏻
---
![GitHub Repo stars](https://img.shields.io/github/stars/modelscope/ClearerVoice-Studio) *Please help our community project. Star on GitHub!*

## News :fire:
- [2024.11] FRCRN speech denoiser has been used more than 2.7 million times on [ModelScope](https://modelscope.cn/models/iic/speech_frcrn_ans_cirm_16k)
- [2024.11] MossFormer speech separator has been used more than 2.5 million times on [ModelScope](https://modelscope.cn/models/iic/speech_mossformer_separation_temporal_8k)

## Repository Structure

### 1. **ClearVoice**  
ClearVoice is a unified inference platform designed for speech processing tasks. It leverages our released pre-trained models (such as [FRCRN](https://arxiv.org/abs/2206.07293), [MossFormer](https://arxiv.org/abs/2302.11824)) for speech enhancement, speech separation, and audio-visual target speaker extraction, all trained on extensive datasets. If you are specifically interested in using the pre-trained models, ClearVoice is the ideal solution. Click on [`ClearVoice`](https://github.com/modelscope/ClearerVoice-Studio/tree/main/clearvoice) and follow our detailed instructions to get started.

More speech processing tasks will be included in ClearVoice.

### 2. **Train**  
For advanced researchers and developers, we provide model finetune and training scripts for all the tasks:

- **[Speech enhancement](train/speech_enhancement)**
- **[Speech separation](train/speech_separation)**
- **[Target speaker extraction](train/target_speaker_extraction)**
  - **Audio-only Speaker Extraction Conditioned on a Reference Speech**
  - **Audio-visual Speaker Extraction Conditioned on Face (Lip) Recording**
  - **Audio-visual Speaker Extraction Conditioned on Body Gestures**
  - **Neuro-steered Speaker Extraction Conditioned on EEG Signals**

### 3. **SpeechScore**  
<a href="https://github.com/modelscope/ClearerVoice-Studio/tree/main/speechscore">`SpeechScore`<a/> is a speech quality assessment toolkit. It can be used to assess the speech quality of your processed audio files and evaluate different model performances. It includes many popular speech metrics:

- Signal-to-Noise Ratio (SNR)
- Perceptual Evaluation of Speech Quality (PESQ)
- Short-Time Objective Intelligibility (STOI)
- Deep Noise Suppression Mean Opinion Score (DNSMOS)
- Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
- and many more quality benchmarks  



## What‘s new :fire:
- [2024.11] Release of this repo
  
## Contact
If you have any comment or question about ClearerVoice-Studio, please contact us by
- email: {shengkui.zhao, zexu.pan}@alibaba-inc.com


## Acknowledge
ClearerVoice-Studio contains third-party components and code modified from some open-source repos, including: <br>
[Speechbrain](https://github.com/speechbrain/speechbrain), [ESPnet](https://github.com/espnet), [TalkNet-ASD
](https://github.com/TaoRuijie/TalkNet-ASD)
