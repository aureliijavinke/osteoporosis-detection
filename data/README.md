# Dataset

The dataset is not included in this repository.

This project expects a binary medical image classification dataset organized into three subsets:

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

Each class folder should contain image files belonging to that category.

The training script uses `torchvision.datasets.ImageFolder`, so the class names are inferred directly from the folder names.

Before running the project, update the dataset path in `src/train.py` if needed:

```python
BASE_PATH = "/content/drive/MyDrive/dataset"
```

Medical image datasets may have licensing, privacy or redistribution restrictions. For that reason, only the project code and documentation are included in this repository.
