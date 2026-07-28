import cv2
import os
import json
import random

def slice_video_dataset(
    raw_dir="data/raw_videos",
    ann_path="data/annotations.json",
    output_dir="data/dataset",
    target_fps=10,
    num_frames=15,
    val_split=0.2
):
    if not os.path.exists(ann_path):
        print(f"Error: Annotations file '{ann_path}' not found.")
        return

    with open(ann_path, "r") as f:
        annotations = json.load(f)

    for vid_name, events in annotations.items():
        vid_path = os.path.join(raw_dir, vid_name)
        if not os.path.exists(vid_path):
            print(f"Skipping {vid_name}: File not found.")
            continue

        cap = cv2.VideoCapture(vid_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        stride = max(1, int(orig_fps / target_fps))

        # 1. Slice Action Subclips
        for i, event in enumerate(events):
            split = "val" if random.random() < val_split else "train"
            save_folder = os.path.join(output_dir, split, event["label"])
            os.makedirs(save_folder, exist_ok=True)

            start_frame = int(event["start"] * orig_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames = []
            curr_frame = 0
            while len(frames) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if curr_frame % stride == 0:
                    frames.append(cv2.resize(frame, (224, 224)))
                curr_frame += 1

            if len(frames) == num_frames:
                out_path = os.path.join(save_folder, f"{os.path.splitext(vid_name)[0]}_act_{i}.mp4")
                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (224, 224))
                for f in frames:
                    out.write(f)
                out.release()

        # 2. Slice Background Subclip from start of video (0.0s)
        bg_folder = os.path.join(output_dir, "train", "background")
        os.makedirs(bg_folder, exist_ok=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        bg_frames = []
        curr_frame = 0
        while len(bg_frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if curr_frame % stride == 0:
                bg_frames.append(cv2.resize(frame, (224, 224)))
            curr_frame += 1

        if len(bg_frames) == num_frames:
            out_path = os.path.join(bg_folder, f"{os.path.splitext(vid_name)[0]}_bg.mp4")
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (224, 224))
            for f in bg_frames:
                out.write(f)
            out.release()

        cap.release()

    print("Dataset slicing complete! Files stored in 'data/dataset/'")

if __name__ == "__main__":
    slice_video_dataset()
