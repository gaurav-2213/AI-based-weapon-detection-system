from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .annotate import draw_detections
from .detector import WeaponDetector


ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Weapon Detection Demo")
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# Initialize the detector once (startup cost is paid a single time).
detector = WeaponDetector(model_path=os.environ.get("MODEL_PATH", "yolov8n.pt"))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/favicon.ico")
def favicon() -> Response:
    # Avoid noisy 404s in the browser if no favicon is provided.
    icon = WEB_DIR / "favicon.ico"
    if icon.exists():
        return FileResponse(str(icon), media_type="image/x-icon")
    return Response(status_code=204)


@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)) -> FileResponse:
    job = uuid.uuid4().hex
    safe_name = Path(file.filename or "upload").name
    inp = RUNS_DIR / f"{job}_{safe_name}"
    outp = RUNS_DIR / f"{job}_annotated.jpg"

    with inp.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        bgr = cv2.imread(str(inp))
        if bgr is None:
            raise ValueError("Unable to read the uploaded image.")

        dets = detector.detect_objects(bgr)
        annotated = draw_detections(bgr, dets)

        ok = cv2.imwrite(str(outp), annotated)
        if not ok:
            raise ValueError("Failed to write annotated image output.")

        return FileResponse(
            str(outp),
            media_type="image/jpeg",
            filename=outp.name,
        )
    except Exception:
        # If we fail after partially writing the output, remove the orphan output too.
        if outp.exists():
            outp.unlink(missing_ok=True)
        raise
    finally:
        # Always delete the uploaded input to avoid orphan files on error.
        if inp.exists():
            inp.unlink(missing_ok=True)


@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)) -> FileResponse:
    job = uuid.uuid4().hex
    safe_name = Path(file.filename or "upload").name
    inp = RUNS_DIR / f"{job}_{safe_name}"
    outp = RUNS_DIR / f"{job}_annotated.mp4"

    with inp.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = None
    writer = None
    try:
        cap = cv2.VideoCapture(str(inp))
        if not cap.isOpened():
            raise ValueError("Unable to open the uploaded video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if w <= 0 or h <= 0:
            raise ValueError("Invalid video dimensions.")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(outp), fourcc, float(fps), (w, h))
        if not writer.isOpened():
            raise ValueError("Failed to create video writer.")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            dets = detector.detect_objects(frame)
            annotated = draw_detections(frame, dets)
            writer.write(annotated)

        return FileResponse(
            str(outp),
            media_type="video/mp4",
            filename=outp.name,
        )
    except Exception:
        if outp.exists():
            outp.unlink(missing_ok=True)
        raise
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        # Always delete uploaded input to prevent orphan files on error.
        if inp.exists():
            inp.unlink(missing_ok=True)

