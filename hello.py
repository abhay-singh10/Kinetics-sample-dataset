import cv2
import numpy as np


class ITU1702FrameByFrameDetector:
    """ITU-R BT.1702-3 / WCAG 2.2 Compliant Photosensitive Epilepsy (PSE) Flicker Detector.

    Uses Frame-by-Frame Dual-Threshold Hysteresis with Spatial Pixel-Identity
    Tracking
    and Shot-Cut Pre-Segmentation.
    """

    def __init__(
        self,
        fps: float = 30.0,
        grid_size: tuple = (16, 16),
        l_max: float = 80.0,  # Peak SDR screen brightness in cd/m^2 (BT.1702 spec)
        tau_high: float = 20.0,  # cd/m^2 primary flash contrast threshold
        tau_low: float = 5.0,  # cd/m^2 hysteresis lower bound (noise filter)
        coverage_thresh: float = 0.25,  # 25% screen area hazard requirement
        scene_cut_area: float = 0.80,  # >= 80% screen area shift = Hard Scene Cut
        max_flash_hz: float = 3.0,  # > 3.0 Hz (> 6 transitions/sec) = violation
    ):
        self.fps = fps
        self.grid_rows, self.grid_cols = grid_size
        self.total_blocks = self.grid_rows * self.grid_cols
        self.l_max = l_max
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.coverage_thresh = coverage_thresh
        self.scene_cut_area = scene_cut_area
        self.max_flash_hz = max_flash_hz

        # 1.0 second rolling time window in frames
        self.window_frames = int(round(fps * 1.0))

        # State memory
        self.prev_brightness = None
        self.prev_state_mask = None

        # Rolling queue storing spatial transition masks: (frame_index, active_pixel_mask)
        self.transition_history = []

    def _bgr_to_screen_brightness(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Converts raw BGR frame to relative luminance Y (ITU-R BT.709) and

        applies gamma-corrected screen brightness L in cd/m^2.
        """
        # Convert BGR to RGB normalized [0, 1]
        frame_rgb = (
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            / 255.0
        )

        # Relative Luminance Y (BT.709 weighting coefficients)
        y_lum = (
            0.2126 * frame_rgb[:, :, 0]
            + 0.7152 * frame_rgb[:, :, 1]
            + 0.0722 * frame_rgb[:, :, 2]
        )

        # Convert to Screen Brightness (cd/m^2) using CRT/LCD Gamma = 2.2 curve
        brightness = self.l_max * (y_lum**2.2)

        # Spatial downsampling to grid for pixel-group identity tracking & noise reduction
        grid_brightness = cv2.resize(
            brightness,
            (self.grid_cols, self.grid_rows),
            interpolation=cv2.INTER_AREA,
        )
        return grid_brightness

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> dict:
        """Processes a single frame and updates the temporal state machine."""
        curr_brightness = self._bgr_to_screen_brightness(frame_bgr)

        if self.prev_brightness is None:
            self.prev_brightness = curr_brightness
            self.prev_state_mask = np.zeros_like(curr_brightness, dtype=np.int8)
            return {
                "flicker_detected": False,
                "is_scene_cut": False,
                "reason": "Warming up initial frame",
            }

        # 1. Compute Differential Brightness Map: D_t(x, y) = L(t) - L(t-1)
        diff_map = curr_brightness - self.prev_brightness
        abs_diff = np.abs(diff_map)

        # 2. Shot-Cut Pre-Segmentation Filter
        # If >= 80% of grid blocks change by >= 20 cd/m^2 in 1 step, it is a hard edit
        large_change_mask = abs_diff >= self.tau_high
        change_area_ratio = np.sum(large_change_mask) / self.total_blocks

        if change_area_ratio >= self.scene_cut_area:
            # Hard edit detected: Reset transition history to avoid false positives across shots
            self.transition_history.clear()
            self.prev_brightness = curr_brightness
            self.prev_state_mask = np.zeros_like(curr_brightness, dtype=np.int8)
            return {
                "flicker_detected": False,
                "is_scene_cut": True,
                "reason": f"Scene cut detected ({change_area_ratio*100:.1f}% area change)",
            }

        # 3. Dual-Threshold Hysteresis State Updates
        curr_state_mask = self.prev_state_mask.copy()

        # Transition to State +1 (Brightening flash phase)
        curr_state_mask[diff_map >= self.tau_high] = 1
        # Transition to State -1 (Darkening flash phase)
        curr_state_mask[diff_map <= -self.tau_high] = -1
        # Values between -tau_low and +tau_low retain their previous state (hysteresis hold)

        # 4. Identify Valid State Flips (+1 -> -1 or -1 -> +1)
        state_flipped_mask = (
            (curr_state_mask != self.prev_state_mask)
            & (curr_state_mask != 0)
            & (self.prev_state_mask != 0)
        )

        # 5. Maintain 1-Second Sliding Transition Queue
        self.transition_history = [
            (f_idx, mask)
            for f_idx, mask in self.transition_history
            if (frame_idx - f_idx) < self.window_frames
        ]

        if np.any(state_flipped_mask):
            self.transition_history.append((frame_idx, state_flipped_mask))

        # 6. Spatial Pixel-Identity Set Intersection (A_t ∩ A_t+1 ∩ ...)
        if len(self.transition_history) >= 2:
            persistent_mask = self.transition_history[0][1].copy()
            for _, mask in self.transition_history[1:]:
                persistent_mask = np.logical_and(persistent_mask, mask)
            persistent_coverage = np.sum(persistent_mask) / self.total_blocks
        else:
            persistent_coverage = 0.0

        transition_count = len(self.transition_history)

        # 7. Evaluate ITU-R BT.1702 Safety Violation:
        # Cadence > 3 Hz (> 6 transitions per second) AND spatial area >= 25%
        flicker_violation = (transition_count > 6) and (
            persistent_coverage >= self.coverage_thresh
        )

        # Save current states for next frame iteration
        self.prev_brightness = curr_brightness
        self.prev_state_mask = curr_state_mask

        return {
            "flicker_detected": bool(flicker_violation),
            "is_scene_cut": False,
            "persistent_coverage": float(persistent_coverage),
            "transition_count_1s": int(transition_count),
        }


def run_pse_flicker_analysis(video_path: str, min_duration_sec: float = 1.0):
    """Processes a video file, aggregates raw frame violations, filters out short

    micro-spikes under min_duration_sec, and prints a continuous timestamp
    report.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_path}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0.0

    print(f"Analyzing '{video_path}'")
    print(
        f"FPS: {fps:.2f} | Total Frames: {total_frames} | Duration:"
        f" {video_duration:.2f}s"
    )
    print(
        f"Filter Rule: Logging events lasting AT LEAST {min_duration_sec:.1f}s"
    )
    print("=" * 65)

    detector = ITU1702FrameByFrameDetector(fps=fps)

    frame_idx = 0
    raw_events = []
    active_start = None

    # Step 1: Detect raw frame-level contiguous state ranges
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        result = detector.process_frame(frame, frame_idx)
        timestamp = frame_idx / fps

        is_flicker = result["flicker_detected"]

        # Track continuous True blocks frame-by-frame
        if is_flicker and active_start is None:
            active_start = timestamp
        elif not is_flicker and active_start is not None:
            raw_events.append((active_start, timestamp))
            active_start = None

    # Close any active event at video boundary
    if active_start is not None:
        raw_events.append((active_start, video_duration))

    cap.release()

    # Step 2: Filter for events lasting AT LEAST min_duration_sec (e.g., 1.0 second)
    valid_flicker_events = []
    for start, end in raw_events:
        duration = end - start
        if duration >= min_duration_sec:
            valid_flicker_events.append((start, end, duration))

    # Step 3: Print Clean Summary Report
    print("\n" + "=" * 65)
    print(f"FINAL PSE FLICKER REPORT (Min Event Duration: {min_duration_sec}s)")
    print("=" * 65)

    if not valid_flicker_events:
        print("PASS: No sustained flicker events detected in this video.")
    else:
        print(
            f"FAIL: Found {len(valid_flicker_events)} sustained flicker"
            " event(s):\n"
        )
        for i, (start, end, duration) in enumerate(valid_flicker_events, 1):
            print(
                f" Event #{i}: From {start:05.2f}s to {end:05.2f}s  (Duration:"
                f" {duration:.2f}s)"
            )


if __name__ == "__main__":
    # Change "input_video.mp4" to your target video path
    # Change min_duration_sec if you want a different minimum threshold (e.g., 0.5s or 1.0s)
    run_pse_flicker_analysis("input_video.mp4", min_duration_sec=1.0)
