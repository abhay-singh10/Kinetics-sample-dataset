import os
import glob
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision import transforms

# Import your stored model architecture
from model import build_movinet_a2_stream, ConvBlock3D


# ==========================================
# 1. Multi-Label Frame-based Dataset
# ==========================================

class FrameStreamDataset(Dataset):
    """
    Dataset that loads pre-cropped 224x224 frame images from clip folders
    and supports multi-label annotations (e.g. 'punch,walk').
    """
    def __init__(self, manifest_csv: str, base_frames_dir: str, class_to_idx: dict, clip_len: int = 15):
        self.df = pd.read_csv(manifest_csv)
        self.base_frames_dir = base_frames_dir
        self.class_to_idx = class_to_idx
        self.num_classes = len(class_to_idx)
        self.clip_len = clip_len

        # Image transformations (Normalized to [0, 1])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), # Automatically scales pixels to [0, 1]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_folder = os.path.join(self.base_frames_dir, str(row['clip_folder']))
        
        # ----------------------------------------------------
        # MULTI-LABEL: Parse comma-separated labels
        # ----------------------------------------------------
        label_str = str(row['label'])
        active_labels = [l.strip() for l in label_str.split(',') if l.strip()]
        
        # Create Float32 multi-hot binary vector [0.0, 1.0, 0.0, ...]
        target_vector = torch.zeros(self.num_classes, dtype=torch.float32)
        for label_name in active_labels:
            if label_name in self.class_to_idx:
                target_vector[self.class_to_idx[label_name]] = 1.0

        # Read image files in sorted numerical order
        frame_paths = sorted(
            glob.glob(os.path.join(clip_folder, "*.jpg")) + 
            glob.glob(os.path.join(clip_folder, "*.png"))
        )

        frames = []
        for p in frame_paths[:self.clip_len]:
            img = Image.open(p).convert('RGB')
            img_tensor = self.transform(img)
            frames.append(img_tensor)

        # Pad clip with zero tensors if it ends shorter than clip_len
        while len(frames) < self.clip_len:
            pad_tensor = torch.zeros(3, 224, 224) if len(frames) == 0 else frames[-1].clone()
            frames.append(pad_tensor)

        # Stack list of (3, 224, 224) tensors into (3, T, 224, 224) for MoViNet
        video_tensor = torch.stack(frames, dim=1)
        
        return video_tensor, target_vector


# ==========================================
# 2. Main Training Execution
# ==========================================

def main():
    # ----------------------------------
    # Configurations & Hyperparameters
    # ----------------------------------
    MANIFEST_CSV = "train_manifest.csv"
    FRAMES_DIR = "dataset/frames"
    BATCH_SIZE = 8
    CLIP_LEN = 15          # 15 frames at 10 FPS = 1.5 seconds clip
    EPOCHS = 15
    LR_BACKBONE = 1e-4     # Learning rate for unfrozen backbone blocks
    LR_HEAD = 1e-3         # Learning rate for classification head
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Parse unique individual classes from potentially comma-separated labels
    df_manifest = pd.read_csv(MANIFEST_CSV)
    raw_labels = df_manifest['label'].dropna().astype(str).tolist()
    
    unique_labels_set = set()
    for row_str in raw_labels:
        for single_label in row_str.split(','):
            cleaned = single_label.strip()
            if cleaned:
                unique_labels_set.add(cleaned)
                
    unique_labels = sorted(list(unique_labels_set))
    class_to_idx = {label_name: idx for idx, label_name in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    
    print(f"Detected {num_classes} classes: {class_to_idx}")

    # ----------------------------------
    # Data Loader Setup
    # ----------------------------------
    train_dataset = FrameStreamDataset(
        manifest_csv=MANIFEST_CSV,
        base_frames_dir=FRAMES_DIR,
        class_to_idx=class_to_idx,
        clip_len=CLIP_LEN
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # ----------------------------------
    # Model Setup & Layer Freezing
    # ----------------------------------
    print("\nInitializing MoViNet-A2 Stream model...")
    model = build_movinet_a2_stream(load_weights=True)

    # Swap final classifier head to match custom dataset class count
    model.classifier[3] = ConvBlock3D(
        in_planes=2048,
        out_planes=num_classes,
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
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # ----------------------------------
    # Optimizer & Loss Function
    # ----------------------------------
    # MULTI-LABEL: BCEWithLogitsLoss treats each class as an independent binary decision
    criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW([
        {'params': [p for name, p in model.named_parameters() if 'classifier' not in name and p.requires_grad], 'lr': LR_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD}
    ], weight_decay=0.01)

    # ----------------------------------
    # Training Loop
    # ----------------------------------
    print("\nStarting multi-label training loop...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        exact_match_correct = 0
        total_samples = 0

        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos = videos.to(device)  # (B, 3, T, 224, 224)
            labels = labels.to(device)  # (B, Num_Classes) Float32 tensor

            # Reset streaming state buffers at the start of every clip batch
            model.clean_activation_buffers()

            optimizer.zero_grad()

            # Forward pass across sequence
            outputs = model(videos)  # Raw unnormalized logits (B, Num_Classes)
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Clean buffers post-backprop
            model.clean_activation_buffers()

            # Metrics
            running_loss += loss.item() * videos.size(0)
            
            # MULTI-LABEL PREDICTIONS: Apply Sigmoid + 0.5 Threshold
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            
            # Exact Match Accuracy: Check if all active classes in sample are predicted correctly
            exact_match_correct += (preds == labels).all(dim=1).sum().item()
            total_samples += videos.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = (exact_match_correct / total_samples) * 100.0

        print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] - Loss: {epoch_loss:.4f} | Exact Match Acc: {epoch_acc:.2f}%")

    # ----------------------------------
    # Save Fine-Tuned Checkpoint
    # ----------------------------------
    save_path = "movinet_a2_stream_multilabel.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel fine-tuning complete! Weights saved to: {save_path}")


if __name__ == "__main__":
    main()
