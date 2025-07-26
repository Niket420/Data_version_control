import os
import json
import random
import logging
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns  # optional

import yaml
from model_tuning import CustomCNN  # keep your existing model definition

# ------------------------- Logging -------------------------
logging.basicConfig(
    filename='process.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ------------------------- Utils -------------------------
def load_params(path: str = "params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_and_save_confusion_matrix(cm: np.ndarray,
                                   class_names,
                                   out_path: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_classwise_accuracy(cm: np.ndarray) -> Dict[str, float]:
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc, nan=0.0)
    return per_class_acc


def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             class_names) -> Dict[str, Any]:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    overall_acc = (all_preds == all_labels).mean() * 100.0
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    classwise_acc = compute_classwise_accuracy(cm)
    cls_report = classification_report(all_labels, all_preds,
                                       target_names=class_names,
                                       output_dict=True)

    return {
        "overall_accuracy": overall_acc,
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": {
            class_names[i]: float(classwise_acc[i] * 100.0)
            for i in range(len(class_names))
        },
        "classification_report": cls_report,
        "y_true": all_labels.tolist(),
        "y_pred": all_preds.tolist()
    }


def main():
    params = load_params()

    # ----- read params -----
    seed = params['version']['seed']
    batch_size = params['parameter']["batch_size"]
    conv_layers = params['parameter']["conv_layers"]
    conv_filters = params['parameter']["conv_filters"]
    kernel_sizes = params['parameter']["kernel_sizes"]

    # allow kernel_sizes to be single int
    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes] * conv_layers

    # data path (use your split output dir)
    data_root = params['version']['output_data']
    test_data_root = os.path.join(data_root, 'test')

    # model artifacts / reports
    weights_path = params.get('evaluate', {}).get('weights_path', 'models/model.pth')
    reports_dir = params.get('evaluate', {}).get('reports_dir', 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_json_path = os.path.join(reports_dir, f"eval_{timestamp}.json")
    cm_png_path = os.path.join(reports_dir, f"confusion_matrix_{timestamp}.png")

    # ----- reproducibility -----
    set_seeds(seed)

    # ----- device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ----- data -----
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    test_ds = ImageFolder(test_data_root, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    class_names = test_ds.classes
    logger.info(f"Loaded test dataset from: {test_data_root}")
    logger.info(f"Number of test samples: {len(test_ds)}")
    logger.info(f"Classes: {class_names}")

    # ----- model -----
    model = CustomCNN(conv_layers, conv_filters, kernel_sizes).to(device)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Trained weights not found at: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    logger.info(f"Loaded model weights from: {weights_path}")

    # ----- evaluate -----
    metrics = evaluate(model, test_loader, device, class_names)

    logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
    logger.info("Per-class accuracy:")
    for cls, acc in metrics['per_class_accuracy'].items():
        logger.info(f"  {cls}: {acc:.2f}%")

    # ----- save artifacts -----
    cm_np = np.array(metrics["confusion_matrix"])
    plot_and_save_confusion_matrix(cm_np, class_names, cm_png_path)
    logger.info(f"Saved confusion matrix to: {cm_png_path}")

    with open(eval_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved evaluation report to: {eval_json_path}")

    print(f"[OK] Evaluation finished.\n"
          f"  -> Report: {eval_json_path}\n"
          f"  -> Confusion Matrix PNG: {cm_png_path}")


if __name__ == "__main__":
    main()