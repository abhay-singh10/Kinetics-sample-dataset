import os
import glob
import random
import logging
import numpy as np
from PIL import Image
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

# Import your stored model architecture
from model import build_movinet_a2_stream, ConvBlock3D


# ==========================================
# 0. Logger Setup
# ==========================================

def setup_logger(log_file="training.log"):
    """
    Sets up a logger that simultaneously prints to the console
    and streams to a persistent 'training.log' file on disk.
    """
    logger = logging.getLogger("MoViNet_Training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Prevent duplicate logging handlers

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Stream to Console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Save to File
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

logger = setup_logger()


# ==========================================
# 1. Label Mapping Configuration
# ==========================================

# Map original 8-bit flag positions (0 to 7) to target consolidated class names.
# Customize this dictionary anytime to merge or ignore labels!
RAW_TO_CONSOLIDATED_MAP = {
    0: "class_A",
    1: "class_A",
    2: "class_B",
    3: "class_B",
    4: "class_C",
    5: "class_C",
    6: "class_D",
    7: "class_D",
}

# Generate unique target classes and index lookups
UNIQUE_TARGET_CLASSES = sorted(list(set(RAW_TO_CONSOLIDATED_MAP.values())))
TARGET_CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(UNIQUE_TARGET_CLASSES)}
NUM_TARGET_CLASSES = len(UNIQUE_TARGET_CLASSES)


def parse_binary_flag_target(folder_name: str) -> np.ndarray:
    """
    Parses an 8-character binary flag string at the end of a subfolder name (e.g., '0_150_11000000')
    and maps it to consolidated target classes using Logical OR.
    """
    binary_flag_str = folder_name.split('_')[-1].strip()
    target_vector = np.zeros(NUM_TARGET_CLASSES, dtype=np.float32)

    for raw_bit_idx, char in enumerate(binary_flag_str):
        if char == '1' and raw_bit_idx in RAW_TO_CONSOLIDATED_MAP:
            target_class_name = RAW_TO_CONSOLIDATED_MAP[raw_bit_idx]
            target_class_idx = TARGET_CLASS_TO_IDX[target_class_name]
            target_vector[target_class_idx] = 1.0  # Logical OR

    return target_vector


# ==========================================
# 2. Clip-Weighted Video-Level Stratified Splitter
# ==========================================

def clip_weighted_multilabel_stratified_split(parent_video_data: dict, test_size: float = 0.20, seed: int = 42):
    """
    Greedy multi-label stratification algorithm operating strictly at the PARENT VIDEO level.
    Guarantees 0% video leakage while balancing target splits by actual estimated CLIP COUNTS
    rather than just video folder counts.
    """
    random.seed(seed)
    np.random.seed(seed)

    parent_video_names = list(parent_video_data.keys())
    random.shuffle(parent_video_names)

    # Compute global total clip counts per class
    total_class_clip_counts = np.zeros(NUM_TARGET_CLASSES, dtype=np.float64)
    for v_info in parent_video_data.values():
        total_class_clip_counts += v_info['clip_counts']

    # Sort parent videos by rarity (videos containing rare active classes assigned first)
    def calculate_rarity(v_name):
        clip_counts = parent_video_data[v_name]['clip_counts']
        active_classes = np.where(clip_counts > 0)[0]
        if len(active_classes) == 0:
            return 0
        return sum(clip_counts[c] / (total_class_clip_counts[c] + 1e-5) for c in active_classes)

    parent_video_names.sort(key=calculate_rarity, reverse=True)

    train_videos, val_videos = [], []
    train_clip_totals = np.zeros(NUM_TARGET_CLASSES, dtype=np.float64)
    val_clip_totals = np.zeros(NUM_TARGET_CLASSES, dtype=np.float64)

    # Calculate total clips per video for capacity tracking
    total_dataset_clips = sum(v_info['total_clips'] for v_info in parent_video_data.values())
    val_target_clip_limit = total_dataset_clips * test_size
    current_val_clips = 0

    for v_name in parent_video_names:
        v_info = parent_video_data[v_name]
        v_clip_counts = v_info['clip_counts']
        v_total_clips = v_info['total_clips']

        # If validation clip quota is reached, assign remaining videos to training
        if current_val_clips >= val_target_clip_limit:
            train_videos.append(v_name)
            train_clip_totals += v_clip_counts
            continue

        train_need = 0.0
        val_need = 0.0

        active_classes = np.where(v_clip_counts > 0)[0]
        for c in active_classes:
            total_c = train_clip_totals[c] + val_clip_totals[c] + 1e-5
            current_val_ratio = val_clip_totals[c] / total_c

            if current_val_ratio < test_size:
                val_need += v_clip_counts[c]
            else:
                train_need += v_clip_counts[c]

        if val_need > train_need and (current_val_clips + v_total_clips) <= (val_target_clip_limit * 1.15):
            val_videos.append(v_name)
            val_clip_totals += v_clip_counts
            current_val_clips += v_total_clips
        else:
            train_videos.append(v_name)
            train_clip_totals += v_clip_counts

    return train_videos, val_videos, train_clip_totals, val_clip_totals


# ==========================================
# 3. Subfolder-Bound Sliding Window Dataset
# ==========================================

class VideoFolderStreamDataset(Dataset):
    """
    Loads pre-cropped 224x224 frame images from assigned parent video subfolders.
    Safely filters out subfolders with <15 frames and generates 15-frame sliding window
    clips strictly within subfolder boundaries.
    """
    def __init__(self, assigned_subfolder_paths: list, clip_len: int = 15, stride: int = 5):
        self.clip_len = clip_len
        self.samples = []  # Stores (list_of_15_frame_paths, target_tensor)

        self.transform = transforms.ToTensor()

        for subfolder_path in assigned_subfolder_paths:
            subfolder_name = os.path.basename(subfolder_path)

            # Parse 8-bit binary flag into target vector
            target_vector_np = parse_binary_flag_target(subfolder_name)
            target_tensor = torch.from_numpy(target_vector_np)

            # Get sorted frame image paths
            frame_paths = sorted(
                glob.glob(os.path.join(subfolder_path, "*.jpg")) + 
                glob.glob(os.path.join(subfolder_path, "*.png"))
            )

            total_frames = len(frame_paths)
            # HARD REQUIREMENT: Ignore subfolders with fewer than 15 frames
            if total_frames < clip_len:
                continue

            # Generate sliding window clips strictly inside this subfolder
            for start_idx in range(0, total_frames - clip_len + 1, stride):
                window_frames = frame_paths[start_idx : start_idx + clip_len]
                self.samples.append((window_frames, target_tensor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, target_tensor = self.samples[idx]

        frames = []
        for p in frame_paths:
            img = Image.open(p).convert('RGB')
            frames.append(self.transform(img))

        # Stack into shape: (3, T=15, 224, 224)
        video_tensor = torch.stack(frames, dim=1)
        return video_tensor, target_tensor


# ==========================================
# 4. Metrics Logging Helper
# ==========================================

def log_per_class_metrics(all_targets: np.ndarray, all_preds: np.ndarray, target_class_to_idx: dict):
    """
    Computes and logs Precision, Recall, and F1-Score for every consolidated class.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )

    logger.info("  --- PER-CLASS VALIDATION BREAKDOWN ---")
    for cls_name, cls_idx in target_class_to_idx.items():
        p, r, f = precision[cls_idx], recall[cls_idx], f1[cls_idx]
        logger.info(f"    Class '{cls_name:<12}': Precision: {p*100:5.2f}% | Recall: {r*100:5.2f}% | F1: {f*100:5.2f}%")


# ==========================================
# 5. Main Training Execution
# ==========================================

def main():
    # ----------------------------------
    # Hyperparameters & L40S Settings
    # ----------------------------------
    DATASET_DIR = "dataset"
    BATCH_SIZE = 32        # Optimized for MoViNet-A2 gradient stability
    CLIP_LEN = 15          # 15 frames at 10 FPS = 1.5 seconds clip
    TRAIN_STRIDE = 5       # Overlapping 5-frame jump for Training (~66% overlap)
    VAL_STRIDE = 15        # Non-overlapping 15-frame jump for Validation
    EPOCHS = 25            # Maximum epoch budget with early stopping & scheduler
    PATIENCE = 5           # Early stopping patience
    LR_BACKBONE = 5e-5     # Conservative learning rate to preserve pre-trained backbone weights
    LR_HEAD = 1.5e-3       # Fast convergence rate for newly initialized 4-class head
    NUM_WORKERS = 8        # CPU threads for fast data loading

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    logger.info(f"Target Consolidated Classes ({NUM_TARGET_CLASSES}): {TARGET_CLASS_TO_IDX}")

    # ----------------------------------
    # 1. Discover Parent Videos & Subfolders
    # ----------------------------------
    # Structure: dataset/Videoname/Subfolder_0_150_flags/
    parent_video_data = {}
    
    logger.info("Scanning dataset hierarchy and estimating subfolder clip weights...")
    for video_name in sorted(os.listdir(DATASET_DIR)):
        video_dir = os.path.join(DATASET_DIR, video_name)
        if not os.path.isdir(video_dir):
            continue

        subfolders_info = []
        video_class_clip_counts = np.zeros(NUM_TARGET_CLASSES, dtype=np.float64)
        video_total_clips = 0

        for subfolder_name in sorted(os.listdir(video_dir)):
            subfolder_path = os.path.join(video_dir, subfolder_name)
            if not os.path.isdir(subfolder_path):
                continue

            # Count valid frame images
            frame_paths = (
                glob.glob(os.path.join(subfolder_path, "*.jpg")) + 
                glob.glob(os.path.join(subfolder_path, "*.png"))
            )
            num_frames = len(frame_paths)

            # Skip subfolders that cannot produce at least one 15-frame clip
            if num_frames < CLIP_LEN:
                continue

            target_vector = parse_binary_flag_target(subfolder_name)
            
            # Estimate number of training clips this subfolder will produce (using TRAIN_STRIDE)
            estimated_clips = (num_frames - CLIP_LEN) // TRAIN_STRIDE + 1
            
            subfolders_info.append({
                'path': subfolder_path,
                'target': target_vector,
                'estimated_clips': estimated_clips
            })

            video_class_clip_counts += (target_vector * estimated_clips)
            video_total_clips += estimated_clips

        if len(subfolders_info) > 0:
            parent_video_data[video_name] = {
                'subfolders': subfolders_info,
                'clip_counts': video_class_clip_counts,
                'total_clips': video_total_clips
            }

    logger.info(f"Discovered {len(parent_video_data)} valid parent videos containing suitable frame segments.")

    # ----------------------------------
    # 2. Parent Video-Level Stratified Split (80/20)
    # ----------------------------------
    train_video_names, val_video_names, train_clip_totals, val_clip_totals = clip_weighted_multilabel_stratified_split(
        parent_video_data, test_size=0.20, seed=42
    )

    # Collect subfolder paths for train and validation
    train_subfolders = []
    for v_name in train_video_names:
        for sf in parent_video_data[v_name]['subfolders']:
            train_subfolders.append(sf['path'])

    val_subfolders = []
    for v_name in val_video_names:
        for sf in parent_video_data[v_name]['subfolders']:
            val_subfolders.append(sf['path'])

    logger.info(f"Parent Videos Split -> Total: {len(parent_video_data)} | Train: {len(train_video_names)} | Val: {len(val_video_names)}")
    logger.info(f"Subfolders Split    -> Train: {len(train_subfolders)} | Val: {len(val_subfolders)}")

    logger.info("Estimated Clip Distribution Check across Video Splits:")
    for cls_name, cls_i in TARGET_CLASS_TO_IDX.items():
        tr_c, va_c = train_clip_totals[cls_i], val_clip_totals[cls_i]
        logger.info(f"  Class '{cls_name}': Train Est Clips={int(tr_c):,} | Val Est Clips={int(va_c):,}")

    # ----------------------------------
    # 3. Create Datasets & Loaders
    # ----------------------------------
    train_dataset = VideoFolderStreamDataset(train_subfolders, clip_len=CLIP_LEN, stride=TRAIN_STRIDE)
    val_dataset = VideoFolderStreamDataset(val_subfolders, clip_len=CLIP_LEN, stride=VAL_STRIDE)

    logger.info(f"Actual 15-frame Clips Generated -> Train (Stride {TRAIN_STRIDE}): {len(train_dataset):,} | Val (Stride {VAL_STRIDE}): {len(val_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ----------------------------------
    # 4. Model Setup & Layer Freezing
    # ----------------------------------
    logger.info("Initializing MoViNet-A2 Stream model...")
    model = build_movinet_a2_stream(load_weights=True)

    # Swap classifier head to match consolidated class count
    model.classifier[3] = ConvBlock3D(
        in_planes=2048,
        out_planes=NUM_TARGET_CLASSES,
        kernel_size=(1, 1, 1),
        tf_like=True,
        causal=True,
        conv_type="2plus1d",
        bias=True
    )

    # 1. Freeze Stem (conv1) and early blocks (Blocks 0 and 1)
    for param in model.conv1.parameters():
        param.requires_grad = False

    for i in [0, 1]:
        for param in model.blocks[i].parameters():
            param.requires_grad = False

    # 2. UNFREEZE Block 2, Block 3, and Block 4
    for i in [2, 3, 4]:
        for param in model.blocks[i].parameters():
            param.requires_grad = True

    # 3. UNFREEZE Head Projection and Classifier
    for param in model.conv7.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(device)

    # Print summary of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # ----------------------------------
    # 5. Optimizer, Loss & Scheduler
    # ----------------------------------
    # Compute dynamic positive weights for BCE loss based on estimated training clip distribution
    total_train_clips = sum(parent_video_data[v]['total_clips'] for v in train_video_names)
    pos_counts = train_clip_totals
    neg_counts = total_train_clips - pos_counts
    pos_weights_np = neg_counts / (pos_counts + 1e-5)
    pos_weights_tensor = torch.tensor(pos_weights_np, dtype=torch.float32).to(device)

    logger.info(f"Dynamic BCE Loss pos_weights: {dict(zip(TARGET_CLASS_TO_IDX.keys(), np.round(pos_weights_np, 2)))}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)

    optimizer = AdamW([
        {'params': [p for name, p in model.named_parameters() if 'classifier' not in name and p.requires_grad], 'lr': LR_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD}
    ], weight_decay=0.01)

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    scaler = GradScaler()
    best_val_loss = float('inf')
    patience_counter = 0

    # ----------------------------------
    # 6. Training & Validation Loop
    # ----------------------------------
    logger.info("Starting multi-label training loop...")
    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_exact_match = 0
        train_total = 0

        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos = videos.to(device)  # (B, 3, T, 224, 224)
            labels = labels.to(device)  # (B, Num_Classes)

            model.clean_activation_buffers()
            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast():
                outputs = model(videos)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            model.clean_activation_buffers()

            train_loss += loss.item() * videos.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            train_exact_match += (preds == labels).all(dim=1).sum().item()
            train_total += videos.size(0)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_exact_match = 0
        val_total = 0

        val_all_targets = []
        val_all_preds = []

        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device)

                model.clean_activation_buffers()
                with autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                model.clean_activation_buffers()

                val_loss += loss.item() * videos.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()

                val_exact_match += (preds == labels).all(dim=1).sum().item()
                val_total += videos.size(0)

                # Store predictions for detailed per-class metric breakdown
                val_all_targets.append(labels.cpu().numpy())
                val_all_preds.append(preds.cpu().numpy())

        # Step Learning Rate Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Process Metrics
        tr_l = train_loss / train_total
        tr_acc = (train_exact_match / train_total) * 100.0
        va_l = val_loss / val_total
        va_acc = (val_exact_match / val_total) * 100.0

        val_all_targets_np = np.vstack(val_all_targets)
        val_all_preds_np = np.vstack(val_all_preds)

        logger.info(
            f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] (LR: {current_lr:.2e}) "
            f"| Train Loss: {tr_l:.4f} (Exact Acc: {tr_acc:.2f}%) "
            f"| Val Loss: {va_l:.4f} (Exact Acc: {va_acc:.2f}%)"
        )

        # Log Precision, Recall, F1 for every class
        log_per_class_metrics(val_all_targets_np, val_all_preds_np, TARGET_CLASS_TO_IDX)

        # --- RICH CHECKPOINT SAVING & EARLY STOPPING ---
        if va_l < best_val_loss:
            best_val_loss = va_l
            patience_counter = 0  # Reset patience on improvement

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': va_l,
                'val_exact_acc': va_acc,
                'class_to_idx': TARGET_CLASS_TO_IDX,
                'raw_to_consolidated_map': RAW_TO_CONSOLIDATED_MAP
            }

            # 1. Main Best Checkpoint File
            best_path = "best_movinet_a2_stream_multilabel.pth"
            torch.save(checkpoint, best_path)

            # 2. Historical Epoch Checkpoint File
            os.makedirs("checkpoints", exist_ok=True)
            epoch_path = f"checkpoints/movinet_epoch_{epoch+1:02d}_valloss_{va_l:.4f}.pth"
            torch.save(checkpoint, epoch_path)

            logger.info(f"  --> Saved NEW BEST checkpoint to '{best_path}' (Val Loss: {va_l:.4f})\n")
        else:
            patience_counter += 1
            logger.info(f"  --> No val loss improvement for {patience_counter}/{PATIENCE} epochs.\n")
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered at Epoch {epoch+1}! Stopping training.")
                break

    logger.info("Training complete! Best weights, metadata, and full logs successfully saved.")


if __name__ == "__main__":
    main()
