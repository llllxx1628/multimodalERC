# Multimodal Emotion Recognition in Conversations

This repository contains the code for a multimodal emotion recognition system for dialogues. The model integrates text, audio, and visual information and is evaluated on two benchmarks, MELD and IEMOCAP. The code provides a complete training pipeline from raw video and audio, through synchronised lip-speech representation learning and feature extraction, to the final emotion classification model.

The codebase is organised so that MELD and IEMOCAP are handled by parallel scripts. The MELD pipeline is driven by `train.sh`, and the IEMOCAP pipeline is driven by `train_iemocap.sh`. Each pipeline has the same two modes: a full mode starting from raw data and a lightweight mode that assumes pre-computed features.

All file names below refer to this repository. Paths that contain dataset locations may need to be adapted to the user environment.

---

## 1. Repository structure

The central Python modules in this project are as follows.

`video_audio_extract.py`  
Extracts face crops and audio segments from the MELD videos. It reads the original MELD mp4 files, detects faces, aligns frames, and saves cropped face sequences and corresponding audio clips in an organised directory layout.

`train_lipsync.py`  
Trains a LipSync-style network that learns to align audio with the visual stream (lip movements). The trained model is later used inside the feature extraction process in order to obtain informative visual and audio embeddings that are consistent across modalities.

`feature_extract.py`  
Loads the preprocessed face sequences and audio signals, applies the LipSync model and other pre-trained encoders such as a Transformer for text and deep networks for audio and visual streams, and writes the resulting features to JSON files. These JSON files are used as input by the final emotion recognition model.

`train_erc.py`  
Trains the multimodal emotion recognition classifier. It reads the feature JSON files, constructs multimodal graph and attention modules, and optimises the model using the training split. The script reports performance on the validation and test splits and saves the best model and the configuration used.

`utils.py`  
Provides shared components used across the scripts, including definition of LipSyncNet, feature computation utilities, model building blocks, and common helper functions.

`multiattn.py`  
Defines the multi-attention fusion module that operates on the text, audio, and visual feature sequences. The module implements the cross-modal attention and self-attention layers that are used in the main classifier.

For IEMOCAP the repository contains parallel scripts, typically named:

- `video_audio_extract_iemocap.py`
- `train_lipsync_iemocap.py`
- `feature_extract_iemocap.py`
- `train_erc_iemocap.py`

These have the same structure as their MELD counterparts but operate on the IEMOCAP data.

---

## 2. Requirements and installation

All required packages are listed in `requirements.txt`.

To create an environment and install dependencies, run:

pip install -r requirements.txt

3. Data preparation
3.1 MELD

Obtain the MELD dataset from the official source and place it on disk. The repository expects access to the original MELD mp4 videos and the corresponding dialogue and utterance metadata. Typical directories include separate folders for train, development, and test splits.

The script video_audio_extract.py contains variables that specify where the MELD mp4 files are located and where the extracted face videos and audio should be written. Before running any training for MELD the user must edit these paths in video_audio_extract.py, train_lipsync.py, and feature_extract.py so that they match the local directory structure.

The feature extraction script writes the final JSON feature files in dedicated folders for train, validation, and test data. The default configuration file for MELD assumes the following feature locations:

Train features in /MELD/Dataset/Data/train_features/

Development features in /MELD/Dataset/Data/dev_features/

Test features in /MELD/Dataset/Data/test_features/

Each directory contains:

text_features.json

audio_features.json

visual_features.json

If the user chooses a different feature directory, the paths in the MELD configuration file must be updated accordingly.

The MELD labels are defined in the configuration file as seven emotion categories: neutral, surprise, fear, sadness, joy, disgust, and anger.

3.2 IEMOCAP

The IEMOCAP pipeline follows the same design. The user must download IEMOCAP, place the wav and video files in suitable directories, and adjust the dataset paths in the IEMOCAP scripts:

video_audio_extract_iemocap.py

train_lipsync_iemocap.py

feature_extract_iemocap.py

train_erc_iemocap.py

The IEMOCAP configuration file (configs/iemocap_config.json) specifies the feature JSON paths for train, validation, and test splits. These files have the same structure as in the MELD case and are consumed by the final classifier.

The IEMOCAP labels are defined in that configuration file and typically include the set of emotions used in the paper, such as neutral, frustration, sadness, anger, excited, and happiness.

4. Configuration files

Each dataset has an associated JSON configuration file in the configs directory.

For MELD, the main configuration file is:

configs/meld_config.json

This file defines the batch size, number of epochs, embedding dimensions for each modality, the set of emotion labels, the feature paths, and the training hyperparameters. A typical configuration includes the following groups of settings.

The batch_size and num_epochs fields control the mini-batch size and the number of passes over the training data. The embed_dims_full block records the dimensionality of the visual, audio, and text features (for example visual 768, audio 1024, text 1024).

The classes block lists the emotion labels for MELD. For public release the configuration only contains the MELD labels, and the pipeline treats MELD as the default dataset.

The feature_paths block defines the locations of the feature JSON files for train, validation, and test splits. For MELD the default values are: