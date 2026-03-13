Skin Cancer Detection using MobileNet and Vision Transformer

**Project Overview**
Developed a deep learning–based skin cancer detection system using dermoscopic images from the HAM10000 dataset.
Implemented and compared MobileNet (CNN) and Vision Transformer (Transformer-based) architectures for skin lesion classification.
The project focuses on automated detection of abnormal skin lesions to support early medical screening.

**Technologies Used**
Python, TensorFlow/Keras
NumPy, Pandas, Matplotlib
Scikit-learn
Google Colab
VS code editor

**Data Processing**
Loaded HAM10000 dataset and metadata.
Removed missing images and performed image resizing (224×224) and normalization.
Applied data augmentation and split the dataset into training, validation, and test sets.

**Models**
MobileNet: Lightweight CNN using depthwise separable convolutions for efficient image classification.
Vision Transformer: Uses patch embeddings and self-attention to capture global image features.
Trained both models and compared their performance.

**Results**
Both models successfully classified multiple skin lesion categories.
Comparison provided insights into CNN vs Transformer performance for medical image classification.

**Future Scope**
Improve accuracy through fine-tuning and larger datasets.
Deploy the model as a real-time web or mobile application for skin screening.
