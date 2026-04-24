# Osteoporosis Detection using Deep Learning

This project focuses on binary classification of medical images into two classes:

- Normal
- Osteoporosis

The model is trained using PyTorch and torchvision. The dataset is organised into train, validation and test folders.

## Project structure

```text
osteoporosis-detection/
├── src/
│   └── train.py
├── results/
│   └── metrics.txt
├── requirements.txt
└── README.md

STRUCTURE
The dataset should be organised as:
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
The dataset itself is not uploaded to GitHub because medical image datasets can be large and may contain sensitive data.

Model
A pretrained ResNet18 model is used and adapted for binary classification.

How to run
Install dependencies:
pip install -r requirements.txt

Run training:
python src/train.py

Results
Training and validation results are saved in the results/ folder.

Author
Aurēlija


