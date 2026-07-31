import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import precision_recall_fscore_support

# Import your model architecture and dataset class from your training code
from model import build_movinet_a2_stream  # Adjust if model function name differs
from dataset import VideoFolderStreamDataset # Adjust if dataset class name differs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEST_DIR = "./data/test"                   # Path to your test dataset directory
CHECKPOINT_PATH = "best_movinet_a2_stream_multilabel.pth"
OUTPUT_DIR = "./test_results"
BATCH_SIZE = 32
CLIP_LEN = 15
TEST_STRIDE = 15                           # Non-overlapping 1.5s evaluation windows
NUM_CLASSES = 4
THRESHOLD = 0.5

CLASS_NAMES = ["Class A", "Class B", "Class C", "Class D"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Setup Logging
# ---------------------------------------------------------------------------
log_file = os.path.join(OUTPUT_DIR, "test_evaluation.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)

# ---------------------------------------------------------------------------
# Main Evaluation Loop
# ---------------------------------------------------------------------------
def run_test():
    logging.info("=" * 60)
    logging.info("Starting MoViNet-A2 Model Evaluation on Test Dataset")
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Loading Checkpoint: {CHECKPOINT_PATH}")
    logging.info(f"Test Directory: {TEST_DIR}")
    logging.info("=" * 60)

    # 1. Load Dataset & DataLoader
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Test directory '{TEST_DIR}' does not exist.")

    test_dataset = VideoFolderStreamDataset(
        root_dir=TEST_DIR,
        clip_len=CLIP_LEN,
        stride=TEST_STRIDE
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    logging.info(f"Loaded {len(test_dataset)} total test clips from {len(test_loader)} batches.")

    # 2. Build Model & Load Saved Weights
    model = build_movinet_a2_stream(num_classes=NUM_CLASSES)
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint file '{CHECKPOINT_PATH}' not found!")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(DEVICE)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    # 3. Inference Loop
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []

    logging.info("\nRunning inference...")
    with torch.no_grad():
        for i, (videos, labels) in enumerate(test_loader):
            videos = videos.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # Clean activation buffers prior to batch processing
            if hasattr(model, "clean_activation_buffers"):
                model.clean_activation_buffers()

            with autocast():
                outputs = model(videos)
                loss = criterion(outputs, labels)

            if hasattr(model, "clean_activation_buffers"):
                model.clean_activation_buffers()

            total_loss += loss.item() * videos.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs >= THRESHOLD).float()

            all_targets.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # 4. Process & Compute Metrics
    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)
    y_prob = np.vstack(all_probs)

    avg_test_loss = total_loss / len(test_dataset)

    # Calculate Exact Match Accuracy (All 4 classes correct simultaneously)
    exact_matches = np.all(y_true == y_pred, axis=1)
    exact_accuracy = np.mean(exact_matches) * 100.0

    # Calculate Per-Class Precision, Recall, and F1 Score
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # 5. Format and Print Results
    logging.info("\n" + "=" * 60)
    logging.info("                  FINAL TEST RESULTS                   ")
    logging.info("=" * 60)
    logging.info(f"Test Loss          : {avg_test_loss:.4f}")
    logging.info(f"Exact Match Acc    : {exact_accuracy:.2f}%")
    logging.info("-" * 60)
    logging.info(f"{'Class':<12} | {'Precision (%)':<15} | {'Recall (%)':<12} | {'F1-Score (%)':<12} | {'Samples':<8}")
    logging.info("-" * 60)

    per_class_summary = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        p_val = precision[idx] * 100
        r_val = recall[idx] * 100
        f1_val = f1[idx] * 100
        sup_val = int(support[idx])

        per_class_summary[class_name] = {
            "precision": round(p_val, 2),
            "recall": round(r_val, 2),
            "f1_score": round(f1_val, 2),
            "support": sup_val
        }

        logging.info(f"{class_name:<12} | {p_val:<15.2f} | {r_val:<12.2f} | {f1_val:<12.2f} | {sup_val:<8}")

    logging.info("=" * 60)

    # 6. Save Clean JSON Report
    json_results = {
        "checkpoint": CHECKPOINT_PATH,
        "test_clips": len(test_dataset),
        "test_loss": round(avg_test_loss, 4),
        "exact_match_accuracy": round(exact_accuracy, 2),
        "per_class_metrics": per_class_summary
    }

    json_path = os.path.join(OUTPUT_DIR, "test_summary.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=4)

    logging.info(f"\n[SUCCESS] Execution complete.")
    logging.info(f"Full text log saved to : {log_file}")
    logging.info(f"Clean JSON summary saved to: {json_path}")


if __name__ == "__main__":
    run_test()
