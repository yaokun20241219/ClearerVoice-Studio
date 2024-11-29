[![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=ClearerVoice-Studio&text2=%20An%20AI-Powered%20Speech%20Processing%20Toolkit&width=1000&height=210)](https://github.com/Akshay090/svg-banners)
    
<strong>ClearerVoice-Studio</strong> is an open-source, AI-powered speech processing toolkit designed for researchers, developers, and end-users. It provides capabilities of speech enhancement, speech separation, target speaker extraction, and more. The toolkit provides state-of-the-art pre-trained models, along with training and inference scripts, all accessible from this repository.

#### 👉🏻[ClearVoice Demo](https://huggingface.co/spaces/alibabasglab/ClearVoice)👈🏻  | 👉🏻[SpeechScore Demo](https://huggingface.co/spaces/alibabasglab/SpeechScore)👈🏻

---
![GitHub Repo stars](https://img.shields.io/github/stars/modelscope/ClearerVoice-Studio) *Please help our community project. Star on GitHub!*

## News :fire:
- [2024.11] FRCRN speech denoiser has been used more than 2.7 million times on [ModelScope](https://modelscope.cn/models/iic/speech_frcrn_ans_cirm_16k)
- [2024.11] MossFormer speech separator has been used more than 2.5 million times on [ModelScope](https://modelscope.cn/models/iic/speech_mossformer_separation_temporal_8k)
- [2024.11] Release of this repository
- Upcoming: More tasks will be added to ClearVoice.
  
## Contents of this repository
This repository is organized into three main components: **[ClearVoice](https://github.com/modelscope/ClearerVoice-Studio/tree/main/clearvoice)**, **[Train](https://github.com/modelscope/ClearerVoice-Studio/tree/main/train)**, and **[SpeechScore](https://github.com/modelscope/ClearerVoice-Studio/tree/main/speechscore)**.

### 1. **ClearVoice**  
ClearVoice offers a user-friendly  solution for speech processing tasks such as speech denoising, separation, audio-visual target speaker extraction, and more. It is designed as a unified inference platform leveraged pre-trained models (e.g., [FRCRN](https://arxiv.org/abs/2206.07293), [MossFormer](https://arxiv.org/abs/2302.11824)), all trained on extensive datasets. If you're looking for a tool to improve speech quality, ClearVoice is the perfect choice. Simply click on [`ClearVoice`](https://github.com/modelscope/ClearerVoice-Studio/tree/main/clearvoice) and follow our detailed instructions to get started.

### 2. **Train**  
For advanced researchers and developers, we provide model finetune and training scripts for all the tasks offerred in ClearVoice and more:

- **Task 1: [Speech enhancement](train/speech_enhancement)** (16kHz & 48kHz)
- **Task 2: [Speech separation](train/speech_separation)** (8kHz & 16kHz)
- **Task 3: [Target speaker extraction](train/target_speaker_extraction)** (16kHz)
  - **Sub-Task 1: Audio-only Speaker Extraction Conditioned on a Reference Speech**
  - **Sub-Task 2: Audio-visual Speaker Extraction Conditioned on Face (Lip) Recording**
  - **Sub-Task 3: Audio-visual Speaker Extraction Conditioned on Body Gestures**
  - **Sub-Task 4: Neuro-steered Speaker Extraction Conditioned on EEG Signals**

Contributors are welcomed to include more model architectures and tasks!

### 3. **SpeechScore**  
<a href="https://github.com/modelscope/ClearerVoice-Studio/tree/main/speechscore">`SpeechScore`<a/> is a speech quality assessment toolkit. We include it here to evaluate different model performance. SpeechScore includes many popular speech metrics:

- Signal-to-Noise Ratio (SNR)
- Perceptual Evaluation of Speech Quality (PESQ)
- Short-Time Objective Intelligibility (STOI)
- Deep Noise Suppression Mean Opinion Score (DNSMOS)
- Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
- and many more quality benchmarks  
  
## Contact
If you have any comments or questions about ClearerVoice-Studio, feel free to raise an issue in this repository or contact us directly at:
- email: {shengkui.zhao, zexu.pan}@alibaba-inc.com


## Acknowledge
ClearerVoice-Studio contains third-party components and code modified from some open-source repos, including: <br>
[Speechbrain](https://github.com/speechbrain/speechbrain), [ESPnet](https://github.com/espnet), [TalkNet-ASD
](https://github.com/TaoRuijie/TalkNet-ASD)
