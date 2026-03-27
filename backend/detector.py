from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    YOLO = None  # type: ignore[assignment]
    _ultralytics_import_error = e


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


WEAPON_KEYWORDS = (
    "knife",
    "gun",
    "pistol",
    "rifle",
    "revolver",
    "shotgun",
    "sword",
    "dagger",
)


class WeaponDetector:
    """
    Weapon-focused wrapper around a general-purpose object detector.

    Notes:
    - By default uses Ultralytics YOLOv8 (auto-downloads weights on first run).
    - Filters detections using WEAPON_KEYWORDS to produce a weapon-only output.
    """

    def __init__(self, model_path: str = "yolov8n.pt", device: Optional[str] = None):
        if YOLO is None:  # pragma: no cover
            raise RuntimeError(
                "ultralytics failed to import. Install dependencies via requirements.txt."
            ) from _ultralytics_import_error

        self.model_path = model_path
        self.device = device
        self._model = YOLO(model_path)
        if device:
            self._model.to(device)

        # Ultralytics exposes class names on the model.
        self._names = self._model.names

    def detect_weapons(
        self,
        bgr_image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> list[Detection]:
        results = self._model.predict(
            source=bgr_image,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        if not results:
            return []

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or boxes.cls is None:
            return []

        out: list[Detection] = []
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy()
        xyxy = boxes.xyxy.detach().cpu().numpy()

        for c, score, b in zip(cls, confs, xyxy):
            label = str(self._names.get(int(c), c))
            if not self._is_weapon_label(label):
                continue
            x1, y1, x2, y2 = [int(v) for v in b.tolist()]
            out.append(
                Detection(
                    label=label,
                    confidence=float(score),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return out

    @staticmethod
    def _is_weapon_label(label: str) -> bool:
        l = label.lower().strip()
        return any(k in l for k in WEAPON_KEYWORDS)

