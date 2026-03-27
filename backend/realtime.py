from __future__ import annotations

import argparse

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

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = detector.detect_weapons(frame, conf=args.conf)
        annotated = draw_detections(frame, dets)
        cv2.imshow("Weapon detection (press q to quit)", annotated)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

