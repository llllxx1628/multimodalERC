# Emotion Recognition in Multi-Speaker Conversations through Speaker Identification, Knowledge Distillation, and Hierarchical Fusion

## Overview

This repository provides the implementation of a multimodal emotion recognition system for dialogue data. The model integrates information from text, audio, and visual modalities, and is evaluated on two benchmark datasets: MELD and IEMOCAP.  
The codebase includes the full pipeline, starting from raw videos and audio, through lip–speech synchronisation learning and feature extraction, and finally training the emotion classification model.

## How to start

#### Clone this repository
```
git clone https://github.com/llllxx1628/multimodalERC.git
```
#### Enviroment setting
```
Hardware: single RTX 3090 GPU, 24GB RAM
conda create -n multimodalERCenv python=3.8  
conda activate MultiEMOEnv
```
#### Install dependencies
```
cd multimoalERC  
pip install -r requirements.txt
```
## Training Options


You can run the project in two ways using the provided train.sh script.


#### Training from scratch


This mode starts from raw MELD videos and audio. It performs all steps automatically.  

Run:  
```
bash train.sh from_scratch
```

#### Training using pre-extracted features


If you already have all features generated, you can directly train the multimodal ERC model without repeating extraction.  

Run:  
```
bash train.sh erc_only
```
