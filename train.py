import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import VideoClipDataset
from model import build_movinet_a2_stream

CONFIG = {
    "num_classes": 3,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "num_epochs": 15,
    "num_frames": 15,
    "train_dir": "data/dataset/train",
    "val_dir": "data/dataset/val",
    "save_path": "checkpoints/best_movinet_a2_stream.pt"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)

        model.clean_activation_buffers()  # Clean buffers before pass
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        model.clean_activation_buffers()  # Clean buffers after step

        running_loss += loss.item() * clips.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    return running_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)

            model.clean_activation_buffers()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            model.clean_activation_buffers()

            running_loss += loss.item() * clips.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    return running_loss / total, correct / total

def main():
    os.makedirs(os.path.dirname(CONFIG["save_path"]), exist_ok=True)
    
    train_ds = VideoClipDataset(CONFIG["train_dir"], CONFIG["num_frames"])
    val_ds = VideoClipDataset(CONFIG["val_dir"], CONFIG["num_frames"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    print(f"Loaded Dataset: {len(train_ds)} train clips | {len(val_ds)} val clips | Classes: {train_ds.classes}")

    model = build_movinet_a2_stream(num_classes=CONFIG["num_classes"]).to(DEVICE)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{CONFIG['num_epochs']:02d}] | Train Acc: {tr_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["save_path"])
            print(f"  --> Saved checkpoint to '{CONFIG['save_path']}'")

if __name__ == "__main__":
    main()
