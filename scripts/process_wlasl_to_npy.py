"""
WLASL Video Processor - extracts 126-dim hand-keypoint .npz files from videos.
Uses MediaPipe Tasks HolisticLandmarker API (hand landmarks only).

Usage:
    python scripts/process_wlasl_to_npy.py --video-dir data/wlasl_videos

Optimizations:
  - Hand landmarks ONLY: 21 left + 21 right = 42 landmarks × 3 = 126 features
  - float16 precision (half memory of float32)
  - np.savez_compressed (compresses 50-70% vs uncompressed .npy)
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import argparse
import sys
from pathlib import Path

SEQUENCE_LENGTH = 30
FRAME_SKIP = 2
MODEL_PATH = "model/holistic_landmarker.task"
TARGET_SIZE = (640, 480)

# Only hand landmarks: 21 × 3 (x, y, z) each
HAND_DIM = 21 * 3  # 63
FEATURE_LENGTH = HAND_DIM * 2  # 126 (left + right)


def extract_hand_keypoints(result):
    """Extract 126-dim hand-only vector from a HolisticLandmarkerResult.

    Returns float16 array: [left_hand_63, right_hand_63] = 126
    Only x, y, z coordinates (no visibility).
    """
    lh = np.array([[lm.x, lm.y, lm.z]
                   for lm in result.left_hand_landmarks], dtype=np.float16).flatten() \
        if result.left_hand_landmarks else np.zeros(HAND_DIM, dtype=np.float16)

    rh = np.array([[lm.x, lm.y, lm.z]
                   for lm in result.right_hand_landmarks], dtype=np.float16).flatten() \
        if result.right_hand_landmarks else np.zeros(HAND_DIM, dtype=np.float16)

    return np.concatenate([lh, rh])


def process_video(video_path, detector):
    """Process a single video and return overlapping 30-frame sequences."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    Could not open: {video_path.name}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < SEQUENCE_LENGTH:
        print(f"    Too short ({total_frames} frames): {video_path.name}")
        cap.release()
        return []

    keypoints_list = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue

        frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = detector.detect(mp_image)
        kp = extract_hand_keypoints(result)
        keypoints_list.append(kp)
        frame_count += 1

    cap.release()

    if len(keypoints_list) < SEQUENCE_LENGTH:
        return []

    # Overlapping sequences with stride 15
    stride = SEQUENCE_LENGTH // 2
    sequences = []
    for i in range(0, len(keypoints_list) - SEQUENCE_LENGTH + 1, stride):
        seq = np.array(keypoints_list[i:i + SEQUENCE_LENGTH], dtype=np.float16)
        sequences.append(seq)

    return sequences



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/wlasl_videos")
    parser.add_argument("--output-dir", default="data/landmarks")
    parser.add_argument("--max-per-sign", type=int, default=150)
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    if not video_dir.exists():
        print(f"Video directory not found: {video_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing MediaPipe Tasks HolisticLandmarker...")
    if not Path(MODEL_PATH).exists():
        print(f"Model not found at {MODEL_PATH}")
        print("Download from: https://storage.googleapis.com/mediapipe-models/"
              "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task")
        sys.exit(1)

    base = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HolisticLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.IMAGE,
    )

    total_sequences = 0

    with vision.HolisticLandmarker.create_from_options(opts) as detector:
        sign_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir()])
        if not sign_dirs:
            print(f"No sign directories found in {video_dir}")
            sys.exit(1)

        for sign_dir in sign_dirs:
            sign_name = sign_dir.name
            sign_output = output_dir / sign_name
            sign_output.mkdir(parents=True, exist_ok=True)

            video_files = sorted(sign_dir.glob("*.mp4")) + sorted(sign_dir.glob("*.webm"))
            if not video_files:
                print(f"\nNo video files in {sign_dir}")
                continue

            print(f"\nProcessing '{sign_name}' ({len(video_files)} videos)...")
            seq_count = 0

            for vf in video_files:
                if seq_count >= args.max_per_sign:
                    break

                try:
                    sequences = process_video(vf, detector)
                except Exception as e:
                    print(f"    Error processing {vf.name}: {e}")
                    continue

                for seq_idx, seq in enumerate(sequences):
                    if seq_count >= args.max_per_sign:
                        break
                    out_path = sign_output / f"{vf.stem}_seq{seq_idx:03d}.npz"
                    # Save compressed with float16
                    np.savez_compressed(out_path, hand=seq)
                    seq_count += 1

            total_sequences += seq_count
            print(f"  => {seq_count} sequences saved")

    print(f"\nDone! {total_sequences} sequences saved to {output_dir.resolve()}")
    print(f"Each .npz[hand] has shape ({SEQUENCE_LENGTH}, {FEATURE_LENGTH}) float16")
    print(f"Feature breakdown: left hand {HAND_DIM} + right hand {HAND_DIM} = {FEATURE_LENGTH}")


if __name__ == "__main__":
    main()