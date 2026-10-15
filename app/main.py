"""视线出屏检测服务。页面在 / ，接口在 /api/gaze/ 。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from app.assets import FACE_FILE, GAZE_FILE, ROOT, ensure_models
from app.engine import GazeEngine
from app.face_detector import YuNetDetector
from app.gaze_estimator import GazeEstimator
from app.geometry import GazeWindow, parse_range
from app.runtime import open_session, provider_label
from app.tracker import GazeTracker

MAX_IMAGE_BYTES = 20 * 1024 * 1024


class Service:
    def __init__(self):
        self.engine: GazeEngine | None = None
        self.provider_text = ""
        self.ready = False
        self.message = "正在加载模型"
        self.error: str | None = None

    def snapshot(self) -> dict:
        return {
            "initialized": self.ready,
            "provider": self.provider_text,
            "message": self.message,
            "error": self.error,
        }


service = Service()


def load_models() -> None:
    ensure_models()
    face_session, face_provider = open_session(FACE_FILE)
    gaze_session, gaze_provider = open_session(GAZE_FILE)
    service.engine = GazeEngine(
        YuNetDetector(face_session),
        GazeEstimator(gaze_session),
        GazeTracker(),
    )
    service.provider_text = provider_label(gaze_provider)
    if face_provider != gaze_provider:
        service.provider_text = (
            f"{provider_label(gaze_provider)} / 人脸 {provider_label(face_provider)}"
        )
    service.ready = True
    service.error = None
    service.message = f"模型已加载（{service.provider_text}）"


def decode_image_bytes(payload: bytes) -> np.ndarray:
    if not payload:
        raise ValueError("图片为空")
    with Image.open(BytesIO(payload)) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("无法解析图片")
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def _window(pitch_range: str, yaw_range: str) -> GazeWindow:
    pitch_min, pitch_max = parse_range(pitch_range)
    yaw_min, yaw_max = parse_range(yaw_range)
    return GazeWindow(pitch_min, pitch_max, yaw_min, yaw_max)


def _detect(payload: bytes, window: GazeWindow, temporal: bool) -> dict:
    if service.engine is None or not service.ready:
        raise RuntimeError(service.error or "模型尚未加载")
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("图片超过 20MB")
    image = decode_image_bytes(payload)
    return service.engine.detect(image, window, temporal)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        await asyncio.to_thread(load_models)
    except Exception as exc:
        service.ready = False
        service.error = str(exc)
        service.message = f"模型加载失败: {exc}"
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/gaze/status")
def status():
    return service.snapshot()


@app.post("/api/gaze/reset")
def reset_camera():
    if service.engine is not None:
        service.engine.reset_camera()
    return {"success": True, "message": "摄像头状态已清空"}


@app.post("/api/gaze/image")
async def detect_image(
    file: UploadFile = File(...),
    pitch_range: str = Form("-30,5"),
    yaw_range: str = Form("-22,22"),
    source: str = Form("image"),
):
    payload = await file.read()
    try:
        window = _window(pitch_range, yaw_range)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    temporal = source == "camera"
    try:
        result = await asyncio.to_thread(_detect, payload, window, temporal)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "success": True,
        "message": result["message"],
        "result": result,
        "provider": service.provider_text,
    }
