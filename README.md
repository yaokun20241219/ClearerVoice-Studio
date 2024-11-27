[![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=ClearerVoice-Studio&text2=%20A%20Speech%20Front-end%20Processing%20Toolkit&width=1000&height=210)](https://github.com/Akshay090/svg-banners)
    
<strong>ClearerVoice-Studio</strong> is an open-source toolkit for **speech enhancement**, **speech separation**, and <a href="https://github.com/modelscope/ClearerVoice-Studio/blob/main/train/target_speaker_extraction">**target speaker extraction**<a/>, for both researchers and developers in speech processing. The repo is organized into three main components: **[ClearVoice](https://github.com/modelscope/ClearerVoice-Studio/tree/main/clearvoice)**, <a href="https://github.com/modelscope/ClearerVoice-Studio/tree/main/speechscore">**SpeechScore**<a/>, and **train**, each tailored to specific needs.

#### 👉🏻[ClearVoice Demo](https://huggingface.co/spaces/alibabasglab/ClearVoice)👈🏻  
#### 👉🏻[SpeechScore Demo](https://huggingface.co/spaces/alibabasglab/SpeechScore)👈🏻
---
Currently, the repo is under updating...

![GitHub Repo stars](https://img.shields.io/github/stars/modelscope/ClearerVoice-Studio) *Please help our community project. Star on GitHub!*

## Repository Structure

### 1. **ClearVoice**  
[`ClearVoice`](https://github.com/modelscope/ClearerVoice-Studio/tree/main/clearvoice) is a unified inference platform. It runs on the pre-trained models for speech enhancement, speech separation, and audio-visual target speaker extraction. We release several pre-trained models that are trained on large datasets. These models can be directly integrated into your projects for speech processing. We plan to include more speech processing tasks in future.

### 2. **train**  
The `train` folder includes scripts and resources to train models for all three tasks:

- **Speech enhancement**
- **Speech separation**
- **[Target speaker extraction](train/target_speaker_extraction)**
  - **Audio-only Speaker Extraction Conditioned on a Reference Speech**
  - **Audio-visual Speaker Extraction Conditioned on Face (Lip) Recording**
  - **Audio-visual Speaker Extraction Conditioned on Body Gestures**
  - **Neuro-steered Speaker Extraction Conditioned on EEG Signals**

### 3. **SpeechScore**  
<a href="https://github.com/modelscope/ClearerVoice-Studio/tree/main/speechscore">`SpeechScore`<a/> is a speech quality assessment toolkit. It includes many popular speech metrics:

- Signal-to-Noise Ratio (SNR)
- Perceptual Evaluation of Speech Quality (PESQ)
- Short-Time Objective Intelligibility (STOI)
- Deep Noise Suppression Mean Opinion Score (DNSMOS)
- Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
- and many more quality benchmarks  

This toolkit can be used to assess the speech quality of your processed audio files and evaluate different model performance. 

## What‘s new :fire:
- [2024.11] Release of this repo
  
## Contact
If you have any comment or question about ClearerVoice-Studio, please contact us by
- email: {shengkui.zhao, zexu.pan}@alibaba-inc.com


## Acknowledge
ClearerVoice-Studio contains third-party components and code modified from some open-source repos, including: <br>
[Speechbrain](https://github.com/speechbrain/speechbrain), [ESPnet](https://github.com/espnet), [TalkNet-ASD
](https://github.com/TaoRuijie/TalkNet-ASD)
