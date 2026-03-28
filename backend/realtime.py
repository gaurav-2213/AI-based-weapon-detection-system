from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .annotate import draw_detections
from .detector import WeaponDetector


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index (e.g. 0) or video path")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    source = int(args.source) if str(args.source).isdigit() else str(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("Could not open video source.")

    detector = WeaponDetector()
    
    # Set up video writer for recording
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0: fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    
    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "realtime_record.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = detector.detect_objects(frame, conf=args.conf)
        annotated = draw_detections(frame, dets)
        
        # Write to recorded file
        if writer.isOpened():
            writer.write(annotated)
            
        cv2.imshow("Weapon detection (press q to quit)", annotated)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    if writer.isOpened():
        writer.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

