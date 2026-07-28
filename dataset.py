import os
import glob
import torch
from torch.utils.data import Dataset
import torchvision.io as io

class VideoClipDataset(Dataset):
    def __init__(self, root_dir, target_frames=15):
        self.clip_paths = glob.glob(os.path.join(root_dir, "*", "*.mp4"))
        self.classes = sorted(list(set([os.path.basename(os.path.dirname(p)) for p in self.clip_paths])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.target_frames = target_frames

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        clip_path = self.clip_paths[idx]
        class_name = os.path.basename(os.path.dirname(clip_path))
        label = self.class_to_idx[class_name]

        # Read video tensor: [Frames, Height, Width, Channels]
        video, _, _ = io.read_video(clip_path, pts_unit='sec')

        # Convert to [Channels, Frames, Height, Width] in range [0, 1]
        video = video.permute(3, 0, 1, 2).float() / 255.0

        # Enforce exact target frame length
        c, t, h, w = video.shape
        if t > self.target_frames:
            video = video[:, :self.target_frames, :, :]
        elif t < self.target_frames:
            pad = torch.zeros((c, self.target_frames - t, h, w))
            video = torch.cat([video, pad], dim=1)

        return video, torch.tensor(label, dtype=torch.long)
