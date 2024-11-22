[![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=ClearerVoice-Studio&text2=%20A%20Speech%20Front-end%20Processing%20Toolkit&width=1000&height=210)](https://github.com/Akshay090/svg-banners)
    
<strong>ClearerVoice-Studio</strong> is an open-source toolkit for **speech enhancement**, **speech separation**, and **target speaker extraction**, for both researchers and developers in speech processing. The repo is organized into three main components: **ClearVoice**, **speechscore**, and **train**, each tailored to specific needs.

---
![GitHub Repo stars](https://img.shields.io/github/stars/modelscope/ClearerVoice-Studio) *Please help our community project. Star on GitHub!*

## Repository Structure

### 1. **clearvoice**  
The `clearvoice` folder contains pre-trained models trained on large datasets. These models are designed for inference and can be directly integrated into your projects.

### 2. **train**  
The `train` folder includes scripts and resources to train models for all three tasks:

- **Speech enhancement**
- **Speech separation**
- **[Target speaker extraction](train/target_speaker_extraction/README.md)**

### 3. **speechscore**  
The `speechscore` folder provides scripts to evaluate the quality of speech processed through the toolkit. It includes metrics and tools to assess:

- Signal-to-Noise Ratio (SNR)
- Perceptual Evaluation of Speech Quality (PESQ)
- Short-Time Objective Intelligibility (STOI)
- Other relevant quality benchmarks  

These scripts help you quantify improvements in your speech processing pipeline.

## What‘s new :fire:
- [2024.11] Release of this repo
  
## Contact
If you have any comment or question about ClearVoice, please contact us by
- email: {shengkui.zhao, zexu.pan}@alibaba-inc.com

## License
ClearVoice is released under the [Apache License 2.0](LICENSE).

## Acknowledge
ClearVoice contains third-party components and code modified from some open-source repos, including: <br>
[Speechbrain](https://github.com/speechbrain/speechbrain), [ESPnet](https://github.com/espnet), [TalkNet-ASD
](https://github.com/TaoRuijie/TalkNet-ASD)
