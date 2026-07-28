import cv2
import torch
import torch.nn.functional as F
from model import build_movinet_a2_stream, load_trained_weights

NUM_CLASSES = 3
CHECKPOINT_PATH = "checkpoints/best_movinet_a2_stream.pt"
CONF_THRESHOLD = 0.65  # Triggers under 1 sec when confidence > 65%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["action_1", "action_2", "background"]

def run_live_stream():
    # Build model skeleton and load fine-tuned checkpoint
    model = build_movinet_a2_stream(num_classes=NUM_CLASSES, freeze_early_stages=False)
    model = load_trained_weights(model, CHECKPOINT_PATH, DEVICE)

    cap = cv2.VideoCapture(0)  # Open default camera
    print("Live Streaming Active... Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess single frame to shape [1, 3, 1, 224, 224]
        resized = cv2.resize(frame, (224, 224))
        frame_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(2).to(DEVICE)

        with torch.no_grad():
            # Model passes 1 frame and retains past state buffers internally
            logits = model(frame_tensor)
            probs = F.softmax(logits, dim=1)[0]

        confidence, pred_idx = torch.max(probs, dim=0)
        pred_label = CLASS_NAMES[pred_idx.item()]

        # Draw UI Overlay
        color = (0, 255, 0) if (confidence.item() > CONF_THRESHOLD and pred_label != "background") else (255, 255, 255)
        text = f"Pred: {pred_label} ({confidence.item()*100:.1f}%)"
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if confidence.item() > CONF_THRESHOLD and pred_label != "background":
            print(f"[ALERT <1s] Detected '{pred_label}' ({confidence.item()*100:.1f}% confidence)")

        cv2.imshow("MoViNet-A2 Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_stream()
