from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .detector import Detection


def draw_detections(bgr: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    img = bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{det.label} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = max(0, y1 - th - baseline - 6)
        cv2.rectangle(img, (x1, y), (x1 + tw + 8, y + th + baseline + 6), (0, 0, 255), -1)
        cv2.putText(
            img,
            label,
            (x1 + 4, y + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return img

