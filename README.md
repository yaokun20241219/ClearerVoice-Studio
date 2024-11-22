# ClearVoice
ClearVoice

This repository provides tools for **speech enhancement**, **speech separation**, and **target speaker extraction**, a resource for research and development in speech processing. The repo is organized into three main components: **ClearVoice**, **speechscore**, and **train**, each tailored to specific needs.

---

## Repository Structure

### 1. **ClearVoice**  
The `ClearVoice` folder contains pre-trained models trained on large datasets. These models are designed for inference and can be directly integrated into your projects.

### 2. **speechscore**  
The `speechscore` folder provides scripts to evaluate the quality of speech processed through the toolkit. It includes metrics and tools to assess:

- Signal-to-Noise Ratio (SNR)
- Perceptual Evaluation of Speech Quality (PESQ)
- Short-Time Objective Intelligibility (STOI)
- Other relevant quality benchmarks  

These scripts help you quantify improvements in your speech processing pipeline.

### 3. **train**  
The `train` folder includes scripts and resources to train models for all three tasks:

- **Speech enhancement**
- **Speech separation**
- **[Target speaker extraction](./train/target_speaker_extraction/README.md)**


