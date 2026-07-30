import os
import glob
import random
import logging
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
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
    0: "punch",       # Bit 0 -> "punch"
    1: "punch",       # Bit 1 -> "punch" (Combines Bit 0 and Bit 1)
    2: "kick",        # Bit 2 -> "kick"
    3: "locomotion",  # Bit 3 -> "locomotion"
    4: "locomotion",  # Bit 4 -> "locomotion" (Combines Bit 3 and Bit 4)
    5: "other",       # Bit 5 -> "other"
    6: "other",       # Bit 6 -> "other"
    7: "other"        # Bit 7 -> "other"
}

# Generate unique target classes and index lookups
UNIQUE_TARGET_CLASSES = sorted(list(set(RAW_TO_CONSOLIDATED_MAP.values())))
TARGET_CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(UNIQUE_TARGET_CLASSES)}
NUM_TARGET_CLASSES = len(UNIQUE_TARGET_CLASSES)


def parse_binary_flag_target(folder_name: str) -> np.ndarray:
    """
    Parses an 8-character binary flag string at the end of a folder name (e.g., '0_150_11000000')
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
# 2. Pure Python Multi-Label Stratified Splitter
# ==========================================

def custom_multilabel_stratified_split(folder_paths: list, folder_targets: np.ndarray, test_size: float = 0.20, seed: int = 42):
    """
    Pure Python multi-label stratification algorithm (Greedy algorithm).
    Splits whole video folders into Train/Val while balancing active class ratios
    without requiring external packages.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    num_folders = len(folder_paths)
    val_target_count = int(num_folders * test_size)
    
    indices = list(range(num_folders))
    random.shuffle(indices)
    
    class_frequencies = folder_targets.sum(axis=0)
    
    # Prioritize folders with rarer classes
    def calculate_rarity(idx):
        active_classes = np.where(folder_targets[idx] == 1)[0]
        if len(active_classes) == 0:
            return 0
        return sum(1.0 / (class_frequencies[c] + 1e-5) for c in active_classes)

    indices.sort(key=calculate_rarity, reverse=True)

    train_idx, val_idx = [], []
    train_counts = np.zeros(folder_targets.shape[1])
    val_counts = np.zeros(folder_targets.shape[1])

    target_val_ratio = test_size

    for idx in indices:
        target = folder_targets[idx]
        active_classes = np.where(target == 1)[0]

        if len(val_idx) >= val_target_count:
            train_idx.append(idx)
            train_counts += target
            continue

        train_need = 0.0
        val_need = 0.0

        for c in active_classes:
            total_c = train_counts[c] + val_counts[c] + 1e-5
            current_val_ratio = val_counts[c] / total_c
            
            if current_val_ratio < target_val_ratio:
                val_need += 1.0
            else:
                train_need += 1.0

        if val_need > train_need and len(val_idx) < val_target_count:
            val_idx.append(idx)
            val_counts += target
        else:
            train_idx.append(idx)
            train_counts += target

    return train_idx, val_idx


# ==========================================
# 3. Folder-Based Sliding Window Dataset
# ==========================================

class VideoFolderStreamDataset(Dataset):
    """
    Loads pre-cropped 224x224 frame images from video folders, parses binary flags,
    and generates 15-frame sliding window clips using configurable strides.
    """
    def __init__(self, folder_paths: list, clip_len: int = 15, stride: int = 5):
        self.clip_len = clip_len
        self.samples = []  # Stores (list_of_15_frame_paths, target_tensor)

        self.transform = transforms.ToTensor()

        for folder_path in folder_paths:
            folder_name = os.path.basename(folder_path)
            
            # Parse 8-bit binary flag into target vector
            target_vector_np = parse_binary_flag_target(folder_name)
            target_tensor = torch.from_numpy(target_vector_np)

            # Get sorted frame image paths
            frame_paths = sorted(
                glob.glob(os.path.join(folder_path, "*.jpg")) + 
                glob.glob(os.path.join(folder_path, "*.png"))
            )

            total_frames = len(frame_paths)
            if total_frames < clip_len:
                continue

            # Generate sliding window clips
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
    BATCH_SIZE = 64        # Optimized for L40S GPU (48GB VRAM)
    CLIP_LEN = 15          # 15 frames at 10 FPS = 1.5 seconds clip
    TRAIN_STRIDE = 5       # Overlapping 5-frame jump for Training
    VAL_STRIDE = 15        # Non-overlapping 15-frame jump for Validation
    EPOCHS = 15
    LR_BACKBONE = 2e-4     # Learning rate for unfrozen backbone blocks
    LR_HEAD = 2e-3         # Learning rate for classification head
    NUM_WORKERS = 8        # CPU threads for fast data loading

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    logger.info(f"Target Consolidated Classes ({NUM_TARGET_CLASSES}): {TARGET_CLASS_TO_IDX}")

    # ----------------------------------
    # 1. Discover Folders & Extract Labels
    # ----------------------------------
    all_folder_paths = []
    folder_targets = []

    for f_name in sorted(os.listdir(DATASET_DIR)):
        f_path = os.path.join(DATASET_DIR, f_name)
        if os.path.isdir(f_path):
            all_folder_paths.append(f_path)
            folder_targets.append(parse_binary_flag_target(f_name))

    all_folder_paths = np.array(all_folder_paths)
    folder_targets = np.array(folder_targets)  # Shape: (Num_Folders, NUM_TARGET_CLASSES)

    # ----------------------------------
    # 2. Pure Python Stratified Split (80/20)
    # ----------------------------------
    train_idx, val_idx = custom_multilabel_stratified_split(
        all_folder_paths, folder_targets, test_size=0.20, seed=42
    )

    train_folders = all_folder_paths[train_idx].tolist()
    val_folders = all_folder_paths[val_idx].tolist()

    logger.info(f"Folders -> Total: {len(all_folder_paths)} | Train: {len(train_folders)} | Val: {len(val_folders)}")

    # Print Stratification Distribution Check
    train_counts = folder_targets[train_idx].sum(axis=0)
    val_counts = folder_targets[val_idx].sum(axis=0)
    logger.info("Class Balance Check across Video Folders:")
    for cls_name, cls_i in TARGET_CLASS_TO_IDX.items():
        tr_c, va_c = train_counts[cls_i], val_counts[cls_i]
        logger.info(f"  Class '{cls_name}': Train Folders={int(tr_c)} | Val Folders={int(va_c)}")

    # ----------------------------------
    # 3. Create Datasets & Loaders
    # ----------------------------------
    train_dataset = VideoFolderStreamDataset(train_folders, clip_len=CLIP_LEN, stride=TRAIN_STRIDE)
    val_dataset = VideoFolderStreamDataset(val_folders, clip_len=CLIP_LEN, stride=VAL_STRIDE)

    logger.info(f"15-frame Clips -> Train (Stride {TRAIN_STRIDE}): {len(train_dataset)} | Val (Stride {VAL_STRIDE}): {len(val_dataset)}")

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
    # 5. Optimizer, Loss & Mixed Precision
    # ----------------------------------
    criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW([
        {'params': [p for name, p in model.named_parameters() if 'classifier' not in name and p.requires_grad], 'lr': LR_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD}
    ], weight_decay=0.01)

    scaler = GradScaler()
    best_val_loss = float('inf')

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

        # Process Metrics
        tr_l = train_loss / train_total
        tr_acc = (train_exact_match / train_total) * 100.0
        va_l = val_loss / val_total
        va_acc = (val_exact_match / val_total) * 100.0

        val_all_targets_np = np.vstack(val_all_targets)
        val_all_preds_np = np.vstack(val_all_preds)

        logger.info(
            f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] "
            f"| Train Loss: {tr_l:.4f} (Exact Acc: {tr_acc:.2f}%) "
            f"| Val Loss: {va_l:.4f} (Exact Acc: {va_acc:.2f}%)"
        )
        
        # Log Precision, Recall, F1 for every class
        log_per_class_metrics(val_all_targets_np, val_all_preds_np, TARGET_CLASS_TO_IDX)

        # --- RICH CHECKPOINT SAVING ---
        if va_l < best_val_loss:
            best_val_loss = va_l

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

    logger.info("Training complete! Best weights, metadata, and full logs successfully saved.")


if __name__ == "__main__":
    main()
