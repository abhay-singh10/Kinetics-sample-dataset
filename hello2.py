import cv2
import numpy as np
import json
import sys
from typing import Dict, List, Tuple, Any

class VideoIllusionDetector:
    """
    Spatiotemporal Optical Flow Analyzer for detecting motion illusions and VIMS triggers
    using sliding 1-second temporal windows with 50% overlap.
    """
    def __init__(
        self,
        video_path: str,
        mag_min_thresh: float = 0.35,
        curl_abs_thresh: float = 0.04,
        curl_ratio_thresh: float = 0.08,
        sign_consistency_thresh: float = 0.75
    ):
        self.video_path = video_path
        self.mag_min_thresh = mag_min_thresh
        self.curl_abs_thresh = curl_abs_thresh
        self.curl_ratio_thresh = curl_ratio_thresh
        self.sign_consistency_thresh = sign_consistency_thresh
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source at path: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 30.0  # Fallback default frame rate
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 1 second window, 50% temporal overlap step
        self.window_frames = max(2, int(round(self.fps * 1.0)))
        self.hop_frames = max(1, int(round(self.fps * 0.5)))
        
        # Pre-compute polar coordinate transformation grids
        cy, cx = self.height / 2.0, self.width / 2.0
        y_grid, x_grid = np.mgrid[0:self.height, 0:self.width]
        dx = x_grid - cx
        dy = y_grid - cy
        
        self.theta = np.arctan2(dy, dx)
        self.sin_theta = np.sin(self.theta)
        self.cos_theta = np.cos(self.theta)
        
        # Define 4 spatial quadrant masks
        half_h, half_w = self.height // 2, self.width // 2
        self.quadrant_masks = [
            (slice(0, half_h), slice(0, half_w)),          # Q1: Top-Left
            (slice(0, half_h), slice(half_w, self.width)),     # Q2: Top-Right
            (slice(half_h, self.height), slice(0, half_w)),    # Q3: Bottom-Left
            (slice(half_h, self.height), slice(half_w, self.width)) # Q4: Bottom-Right
        ]

    def _compute_frame_pair_metrics(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> Dict[str, float]:
        """
        Calculates dense Farnebäck flow and vector calculus properties between contiguous frame pairs.
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        u = flow[..., 0]
        v = flow[..., 1]
        
        # Flow magnitude
        mag = np.hypot(u, v)
        mean_mag = float(np.mean(mag))
        
        # Spatial curl calculation: dv/dx - du/dy
        dv_dx = np.gradient(v, axis=1)
        du_dy = np.gradient(u, axis=0)
        curl = dv_dx - du_dy
        mean_abs_curl = float(np.mean(np.abs(curl)))
        
        # Polar velocity transform: tangential component v_theta
        v_theta = -u * self.sin_theta + v * self.cos_theta
        
        # Check Quadrant Sign Consistency across 4 spatial regions
        q_means = []
        for q_slice in self.quadrant_masks:
            q_val = float(np.mean(v_theta[q_slice]))
            q_means.append(q_val)
            
        # Check if all 4 quadrant means share the same non-zero sign
        q_signs = [np.sign(qm) for qm in q_means]
        has_sign_consistency = (
            all(s == q_signs[0] for s in q_signs) and (q_signs[0] != 0)
        )
        
        return {
            "mean_mag": mean_mag,
            "mean_abs_curl": mean_abs_curl,
            "sign_consistency": 1.0 if has_sign_consistency else 0.0
        }

    def analyze_video(self) -> List[Dict[str, Any]]:
        """
        Processes video stream using sliding 1-second windows with 50% temporal overlap.
        Returns tabular window verdicts and detailed vector field metrics.
        """
        frame_pair_metrics: List[Dict[str, float]] = []
        
        ret, prev_frame = self.cap.read()
        if not ret:
            print("Error: Empty video file.")
            return []
            
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_index = 0
        
        # Step 1: Extract pair-wise metrics across all contiguous frames
        while True:
            ret, curr_frame = self.cap.read()
            if not ret:
                break
                
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            pair_metrics = self._compute_frame_pair_metrics(prev_gray, curr_gray)
            pair_metrics["frame_idx"] = frame_index
            frame_pair_metrics.append(pair_metrics)
            
            prev_gray = curr_gray
            frame_index += 1
            
        self.cap.release()
        
        total_pairs = len(frame_pair_metrics)
        window_verdicts: List[Dict[str, Any]] = []
        
        # Step 2: Evaluate 1-second windows with 50% overlap step
        window_id = 0
        start_pair_idx = 0
        
        while start_pair_idx < total_pairs:
            end_pair_idx = start_pair_idx + (self.window_frames - 1)
            if end_pair_idx > total_pairs:
                end_pair_idx = total_pairs
                
            window_data = frame_pair_metrics[start_pair_idx:end_pair_idx]
            if not window_data:
                break
                
            # Aggregate temporal statistics over the 1-second window
            win_mags = [d["mean_mag"] for d in window_data]
            win_curls = [d["mean_abs_curl"] for d in window_data]
            win_signs = [d["sign_consistency"] for d in window_data]
            
            avg_mag = float(np.mean(win_mags))
            avg_curl = float(np.mean(win_curls))
            sign_consistency_ratio = float(np.mean(win_signs))
            curl_to_mag_ratio = avg_curl / (avg_mag + 1e-6)
            
            start_sec = start_pair_idx / self.fps
            end_sec = end_pair_idx / self.fps
            
            # Step 3: False-positive suppression decision logic
            if avg_mag < self.mag_min_thresh:
                verdict = "NORMAL_STATIC"
                is_illusion = False
                explanation = "Flow magnitude below baseline motion noise floor."
            elif (curl_to_mag_ratio < self.curl_ratio_thresh) or (sign_consistency_ratio < self.sign_consistency_thresh):
                verdict = "NORMAL_TRANSLATION"
                is_illusion = False
                explanation = "Uniform motion field detected (e.g., standard camera pan/tilt); sign consistency failed."
            elif (avg_curl >= self.curl_abs_thresh) and (sign_consistency_ratio >= self.sign_consistency_thresh):
                verdict = "ILLUSORY_ROTATION"
                is_illusion = True
                explanation = "High spatial curl combined with consistent 4-quadrant polar sign coherence detected."
            else:
                verdict = "NORMAL_UNCOHERENT"
                is_illusion = False
                explanation = "Motion field lacks structural rotational coherence."

            window_verdicts.append({
                "window_id": window_id,
                "time_span": f"{start_sec:.2f}s - {end_sec:.2f}s",
                "start_frame": start_pair_idx,
                "end_frame": end_pair_idx,
                "metrics": {
                    "avg_magnitude": round(avg_mag, 4),
                    "avg_abs_curl": round(avg_curl, 4),
                    "curl_mag_ratio": round(curl_to_mag_ratio, 4),
                    "sign_consistency_ratio": round(sign_consistency_ratio, 4)
                },
                "is_illusion": is_illusion,
                "verdict": verdict,
                "explanation": explanation
            })
            
            window_id += 1
            start_pair_idx += self.hop_frames
            
        return window_verdicts

if __name__ == "__main__":
    video_file = "sample_input.mp4"
    
    try:
        detector = VideoIllusionDetector(
            video_path=video_file,
            mag_min_thresh=0.35,
            curl_abs_thresh=0.04,
            curl_ratio_thresh=0.08,
            sign_consistency_thresh=0.75
        )
        results = detector.analyze_video()
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Execution failed: {str(e)}", file=sys.stderr)
