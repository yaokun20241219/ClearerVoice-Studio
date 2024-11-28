# ClearerVoice-Studio: Train Speech Enhancement Models

## 1. Introduction

This repository provides training scripts for speech enhancement models. Currently, it supports fresh train or finetune for the following models:

|model name| sampling rate | Paper Link|
|----------|---------------|------------|
|FRCRN_SE_16K|16000        | FRCRN ([Paper](https://arxiv.org/abs/2206.07293))   |
|MossFormerGAN_SE_16K|16000| MossFormer2 Backbone + GAN ([Paper](https://arxiv.org/abs/2312.11825))|
|MossFormer2_SE_48K  |48000| MossFormer2 Backbone + Masking ([Paper](https://arxiv.org/abs/2312.11825))|

1. **FRCRN_SE_16K**

FRCRN uses a complex network for single-channel speech enhancement. It is a generalized method for enhancing speech in various noise environments. Our trained FRCRN model has won good performance in IEEE ICASSP 2022 DNS Challenge. Please check our [ICASSP paper](https://arxiv.org/abs/2206.07293). 

The FRCRN model is developed based on a new framework of **Convolutional Recurrent Encoder-Decoder (CRED)**, which is built on the Convolutional Encoder-Decoder (CED) architecture. CRED can significantly improve the performance of the convolution kernel by improving the limited receptive fields in the convolutions of CED using frequency recurrent layers. In addition, we introduce the Complex Feedforward Sequential Memory Network (CFSMN) to reduce the complexity of the recurrent network, and apply complex-valued network operations to realize the full complex deep model, which not only constructs long sequence speech more effectively, but also can enhance the amplitude and phase of speech at the same time. 

![model](https://user-images.githubusercontent.com/62317780/203685825-1c349023-c926-45cd-8630-e6289b4d16bd.png)

2. **MossFormerGAN_SE_16K**

MossFormerGAN is motivated from [CMGAN](https://arxiv.org/abs/2203.15149) and [TF-GridNet](https://arxiv.org/abs/2209.03952). We use an extended MossFormer2 backbone (See below figure) to replace Conformer in CMGAN and add the Full-band Self-attention Modul proposed in TF-GridNet. The whole speech enhancemnt network is optimized by the adversarial training scheme as described in CMGAN. We extended the CNN network to an attention-based network for the discriminator. MossFormerGAN is trained for 16kHz speech enhancement.


3. **MossFormer2_SE_48K**



## 2. Usage

### Step-by-Step Guide

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

3. **Download Dataset**
If you don't have any training dataset to start with, we recommend you to download the VoiceBank-DEMAND dataset ([link](https://datashare.ed.ac.uk/handle/10283/2826)]. You may store the dataset anywhere. What you need to start the model training is to create two scp files as shown in `data/tr_demand_28_spks_16k.scp` and `data/cv_demand_testset_16k.scp`. `data/tr_demand_28_spks_16k.scp` contains the training data list and `data/cv_demand_testset_16k.scp` contains the testing data list.

Replace `data/tr_demand_28_spks_16k.scp` and `data/cv_demand_testset_16k.scp` with your new .scp files in `config/train/*.yaml`. Now it is ready to train the models.

4. **Start Training**

``` sh
bash train.sh
```

You may need to set the correct network in `train.sh` and choose if start a fresh training or a finetune process using:
```
network=
train_from_last_checkpoint=
init_checkpoint_path=
```
