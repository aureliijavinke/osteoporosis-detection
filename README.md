# Osteoporosis Detection using Deep Learning

## Overview

This repository demonstrates a binary medical image classification workflow for osteoporosis detection using deep learning.

The project classifies medical images into two classes:

* Normal
* Osteoporosis

The model is implemented using PyTorch and torchvision.

## Project goal

The goal of this project is to demonstrate how to:

* organize a medical image dataset for binary classification;
* load images using `torchvision.datasets.ImageFolder`;
* apply image preprocessing and augmentation;
* fine-tune a pretrained convolutional neural network;
* evaluate model performance using classification metrics;
* save training results and visualizations.

## Technologies used

* Python
* PyTorch
* torchvision
* NumPy
* Matplotlib
* scikit-learn
* Pillow

## Dataset structure

The dataset should be organized into train, validation and test folders:

```text
dataset/
├── train/
│   ├── Normal/
│   └── Osteoporosis/
├── val/
│   ├── Normal/
│   └── Osteoporosis/
└── test/
    ├── Normal/
    └── Osteoporosis/
```

The dataset itself is not included in this repository because medical image datasets may be large and may have usage restrictions.

## Model

The model is based on a pretrained ResNet18 architecture.

The final fully connected layer is replaced with a new output layer for binary classification:

```python
model.fc = nn.Linear(num_features, 2)
```

The two output classes are:

* Normal
* Osteoporosis

## How to run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run training and evaluation:

```bash
python src/train.py
```

Before running the script, update the dataset path in `src/train.py` if needed:

```python
BASE_PATH = "/content/drive/MyDrive/dataset"
```

## Results

Training and evaluation outputs are saved in the `results/` folder.

The script saves:

* best model weights;
* classification report;
* confusion matrix;
* training loss curve;
* validation accuracy curve.

## Limitations

This project is a learning and portfolio demonstration, not a clinically validated diagnostic system.

Prediction quality depends on dataset size, image quality, class balance, preprocessing and model training. The model should not be used for clinical decision-making without proper validation on independent medical datasets.

## What this project demonstrates

This project demonstrates experience with:

* medical image classification;
* PyTorch model training;
* transfer learning with ResNet18;
* image preprocessing and augmentation;
* binary classification evaluation;
* confusion matrix and classification report generation.

## Author

Aurēlija Viņķe


