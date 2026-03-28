from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .detector import Detection


def draw_detections(bgr: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    img = bgr.copy()
    dets_list = list(detections)
    
    people_count = sum(1 for d in dets_list if getattr(d, 'is_person', False))
    weapon_alert = any(getattr(d, 'is_weapon', False) for d in dets_list)
    
    for det in dets_list:
        x1, y1, x2, y2 = det.xyxy
        
        # Default Green
        color = (0, 255, 0)
        label_text = f"{det.label} {det.confidence:.2f}"
        
        if getattr(det, 'is_weapon', False):
            color = (0, 0, 255) # Red for weapon
            label_text = f"ALERT: {det.label} {det.confidence:.2f}"
        elif getattr(det, 'is_person', False):
            color = (255, 0, 0) # Blue for person
            
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = max(0, y1 - th - baseline - 6)
        cv2.rectangle(img, (x1, y), (x1 + tw + 8, y + th + baseline + 6), color, -1)
        cv2.putText(
            img,
            label_text,
            (x1 + 4, y + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if people_count > 0:
        cv2.putText(img, f"People Count: {people_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    if weapon_alert:
        cv2.putText(img, "HIGH ALERT: WEAPON DETECTED", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return img

