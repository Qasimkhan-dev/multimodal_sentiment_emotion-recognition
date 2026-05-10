# Code Folder

This folder contains the complete implementation code for the Multimodal Emotion Recognition and Sentiment Analysis project.

The code is organized into separate scripts for dataset preprocessing, unimodal model training, feature extraction, and multimodal fusion. Each script is responsible for a specific stage of the pipeline, starting from creating balanced MELD dataset subsets to training individual modality models and finally combining extracted features for fusion-based prediction.

The folder includes code for:

- Creating balanced emotion and sentiment datasets
- Extracting audio from MELD video files
- Training text models using RoBERTa-base
- Training audio models using Wav2Vec2-base
- Training video models using MobileNetV2 + Transformer
- Training face models using InceptionResNetV1 + BiLSTM with Attention
- Extracting and saving feature embeddings
- Training 3-modality fusion models
- Training 4-modality fusion models
- Saving checkpoints, features, and evaluation metrics

These scripts support both major tasks of the project: emotion recognition with 7 classes and sentiment analysis with 3 classes. The code follows a step-by-step pipeline where unimodal models are trained first, their features are saved, and then those features are used by fusion models for final prediction.

This folder is the main implementation part of the repository and should be executed according to the training order described in the README.
