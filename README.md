# Colorectal Polyp Classification

This project is a comprehensive machine learning pipeline designed for classifying colorectal polyp images. It incorporates data augmentation, feature extraction using DenseNet121, and classification with XGBoost. The goal is to achieve high accuracy and robustness in the classification task, even with imbalanced datasets.

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Extraction](#feature-extraction)
6. [Model Training](#model-training)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Usage](#usage)
10. [Acknowledgments](#acknowledgments)

---

## Overview

Colorectal polyps are abnormal growths in the colon that can potentially lead to cancer. Early detection and classification of these polyps are critical for effective treatment. This project employs a combination of deep learning and traditional machine learning techniques to classify colorectal polyp images into two classes.

Key Features:
- **Data Augmentation**: Ensures a balanced dataset by generating additional images.
- **Feature Extraction**: Uses DenseNet121 for feature extraction.
- **Classification**: Implements XGBoost with hyperparameter tuning and class balancing.

---

## Project Structure

```
.
├── data2/                     # Original dataset
├── augmented_data/            # Augmented dataset
├── scripts/                   # Python scripts for preprocessing and training
├── xgboost_colorectal_polyp_classifier_augmented.pkl  # Saved XGBoost model
└── README.md                  # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- XGBoost
- scikit-learn
- imbalanced-learn

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/colorectal-polyp-classification.git
   cd colorectal-polyp-classification
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset in the `data2/` directory.

---

## Data Preprocessing

### Augmentation
The dataset is augmented to ensure class balance and improve model generalization. Augmentation techniques include:
- Rotation
- Width and height shifting
- Shearing
- Zooming
- Horizontal and vertical flipping
- Brightness adjustment

To augment the dataset:
```python
augment_data(original_data_dir, augmented_data_dir, target_count=4500)
```

### Splitting
The dataset is split into training, validation, and test sets using `ImageDataGenerator`.

---

## Feature Extraction

Features are extracted from the augmented dataset using DenseNet121, a pre-trained convolutional neural network:
```python
feature_extractor = tf.keras.Model(
    inputs=base_model.input,
    outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
)
```
Extracted features are then used as input for the classifier.

---

## Model Training

The XGBoost classifier is trained with the extracted features. To address class imbalance, SMOTE (Synthetic Minority Oversampling Technique) is applied to the training data.

Key hyperparameters for XGBoost:
- `learning_rate=0.03`
- `max_depth=10`
- `n_estimators=300`
- `scale_pos_weight` for class balancing

Training script:
```python
xgb_model = train_xgboost(train_features_resampled, train_labels_resampled)
```

---

## Evaluation

The model is evaluated using the following metrics:
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1 Score
- AUC-ROC

Example:
```python
evaluate_model(xgb_model, val_features, val_labels)
```

---

## Results

### Validation Set:
- Accuracy: 0.94
- Precision: 0.94
- Recall: 0.94
- Specificity: 0.93
- F1 Score: 0.94
- AUC: 0.98
- Confusion Matrix:
  ```
  [[630  44]
   [ 47 720]]
  ```

### Test Set:
- Accuracy: 0.92
- Precision: 0.94
- Recall: 0.92
- Specificity: 0.93
- F1 Score: 0.93
- AUC: 0.98
- Confusion Matrix:
  ```
  [[383  30]
   [ 39 449]]
  ```

---

## Usage

1. Train the model:
   ```bash
   python train.py
   ```

2. Evaluate the model:
   ```bash
   python evaluate.py
   ```

3. Use the saved model for inference:
   ```python
   from joblib import load
   model = load("xgboost_colorectal_polyp_classifier_augmented.pkl")
   ```

---

## Acknowledgments

This project leverages the DenseNet121 architecture and XGBoost for state-of-the-art performance. Special thanks to the open-source community for providing tools and frameworks that made this project possible.

