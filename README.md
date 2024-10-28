# Undergrad Skin Lesion Classifier

This is an undergraduate thesis project focused on developing a deep learning-based skin lesion classification system. The goal is to create a model that can accurately identify different types of skin lesions, such as melanoma, nevus, and other categories, based on input images.

---

## Project Overview
The project utilizes a modular and configurable design, with the following key components:

- **Data Preprocessing**: Includes a hair removal technique to enhance the quality of the input skin lesion images.
- **Feature Extraction**: Leverages pre-trained EfficientNet models to extract discriminative features from the images.
- **Classification Models**: Implements both a 1D Convolutional Neural Network (CNN) and a Random Forest Classifier for the final classification task.
- **Training Pipeline**: Handles data splitting, class balancing (SMOTE), model training, and performance evaluation.
- **Visualization and Metrics**: Provides utilities for plotting training history and generating classification reports.

The project is designed to be modular and easily configurable, allowing for flexibility in experimenting with different components and hyperparameters.

---

## Features
- Robust data preprocessing, including hair removal
- Efficient feature extraction using EfficientNet models
- Support for multiple classification models (1D CNN, Random Forest)
- Automated class balancing using SMOTE
- Comprehensive training pipeline with callbacks
- Detailed performance visualization and evaluation metrics

---
