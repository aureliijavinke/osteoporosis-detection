import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# =========================

# Configuration

# =========================

BASE_PATH = os.getenv("DATASET_PATH", "/content/drive/MyDrive/dataset")
RESULTS_DIR = Path("results")

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================

# Utility functions

# =========================

def set_seed(seed: int) -> None:
"""Set random seed for reproducible experiments."""
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def create_data_loaders(base_path: str):
"""Create train, validation and test data loaders."""

```
train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)

train_data = datasets.ImageFolder(
    root=os.path.join(base_path, "train"),
    transform=train_transform,
)

val_data = datasets.ImageFolder(
    root=os.path.join(base_path, "val"),
    transform=eval_transform,
)

test_data = datasets.ImageFolder(
    root=os.path.join(base_path, "test"),
    transform=eval_transform,
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
)

return train_data, val_data, test_data, train_loader, val_loader, test_loader
```

def create_model(num_classes: int):
"""Create a pretrained ResNet18 model for image classification."""

```
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

return model.to(DEVICE)
```

def train_model(model, train_loader, val_loader, train_size, val_size):
"""Train the model and keep the best validation checkpoint."""

```
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_model_weights = copy.deepcopy(model.state_dict())
best_val_accuracy = 0.0

train_losses = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / train_size
    train_losses.append(epoch_loss)

    model.eval()
    correct_predictions = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            correct_predictions += (predictions == labels).sum().item()

    val_accuracy = correct_predictions / val_size
    val_accuracies.append(val_accuracy)

    print(f"Training loss: {epoch_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_weights = copy.deepcopy(model.state_dict())

model.load_state_dict(best_model_weights)

return model, best_val_accuracy, train_losses, val_accuracies
```

def evaluate_model(model, test_loader, class_names):
"""Evaluate the trained model on the test set."""

```
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
)

matrix = confusion_matrix(all_labels, all_predictions)

return report, matrix
```

def save_results(
model,
best_val_accuracy,
train_losses,
val_accuracies,
report,
matrix,
class_names,
):
"""Save model weights, metrics and plots."""

```
RESULTS_DIR.mkdir(exist_ok=True)

torch.save(model.state_dict(), RESULTS_DIR / "best_model.pth")

with open(RESULTS_DIR / "metrics.txt", "w", encoding="utf-8") as file:
    file.write("Osteoporosis Detection Results\n")
    file.write("==============================\n\n")
    file.write(f"Best validation accuracy: {best_val_accuracy:.4f}\n\n")
    file.write("Classification report:\n")
    file.write(report)
    file.write("\nConfusion matrix:\n")
    file.write(str(matrix))

plt.figure()
plt.plot(train_losses, label="Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.savefig(RESULTS_DIR / "training_loss.png", bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(val_accuracies, label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.savefig(RESULTS_DIR / "validation_accuracy.png", bbox_inches="tight")
plt.close()

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names,
)

display.plot(values_format="d")
plt.title("Confusion Matrix")
plt.savefig(RESULTS_DIR / "confusion_matrix.png", bbox_inches="tight")
plt.close()
```

def main():
set_seed(RANDOM_SEED)

```
print(f"Using device: {DEVICE}")
print(f"Dataset path: {BASE_PATH}")

(
    train_data,
    val_data,
    test_data,
    train_loader,
    val_loader,
    test_loader,
) = create_data_loaders(BASE_PATH)

class_names = train_data.classes
num_classes = len(class_names)

print(f"Classes: {class_names}")
print(f"Train images: {len(train_data)}")
print(f"Validation images: {len(val_data)}")
print(f"Test images: {len(test_data)}")

model = create_model(num_classes=num_classes)

model, best_val_accuracy, train_losses, val_accuracies = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    train_size=len(train_data),
    val_size=len(val_data),
)

report, matrix = evaluate_model(
    model=model,
    test_loader=test_loader,
    class_names=class_names,
)

print("\nBest validation accuracy:")
print(f"{best_val_accuracy:.4f}")

print("\nClassification report:")
print(report)

print("\nConfusion matrix:")
print(matrix)

save_results(
    model=model,
    best_val_accuracy=best_val_accuracy,
    train_losses=train_losses,
    val_accuracies=val_accuracies,
    report=report,
    matrix=matrix,
    class_names=class_names,
)

print("\nResults saved in the 'results' folder.")
```

if **name** == "**main**":
main()

