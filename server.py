"""
Project Alpha – Web Server
Run:  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import cv2
import threading
import queue
import time
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── Try to import YOLO (graceful fallback for testing without models) ──
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics not installed — running in demo mode (no detection)")

app = FastAPI(title="Project Alpha")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Config ─────────────────────────────────────────────────────
BASE             = Path.home() / "project_alpha"
INFERENCE_WIDTH  = 416
FRAME_SKIP       = 2

LABELS = {
    "person":         ("PERSON",                    (0,   255,   0)),
    "bird":           ("ANIMAL - Bird",             (200, 150,   0)),
    "cat":            ("ANIMAL - Cat",              (200, 150,   0)),
    "dog":            ("ANIMAL - Dog",              (200, 150,   0)),
    "bottle":         ("PLASTIC - Bottle",          (0,   150, 255)),
    "cup":            ("PLASTIC - Cup",             (0,   150, 255)),
    "wine glass":     ("GLASS - Wine glass",        (180, 230, 255)),
    "fork":           ("METAL - Fork",              (180, 180, 180)),
    "knife":          ("METAL - Knife",             (180, 180, 180)),
    "spoon":          ("METAL - Spoon",             (180, 180, 180)),
    "book":           ("PAPER - Book",              (255, 220,   0)),
    "backpack":       ("TEXTILE - Backpack",        (255,   0, 128)),
    "tv":             ("E-WASTE - TV",              (255,   0, 255)),
    "laptop":         ("E-WASTE - Laptop",          (255,   0, 255)),
    "cell phone":     ("PLASTIC - Phone",           (0,   150, 255)),
    "banana":         ("ORGANIC - Banana",          (0,   200,   0)),
    "apple":          ("ORGANIC - Apple",           (0,   200,   0)),
    "chair":          ("BULKY WASTE - Chair",       (150, 100,  50)),
    "couch":          ("BULKY WASTE - Couch",       (150, 100,  50)),
    "microwave":      ("APPLIANCE - Microwave",     (200, 100, 200)),
    "refrigerator":   ("APPLIANCE - Fridge",        (200, 100, 200)),
}

WASTE_COLORS = {
    "METAL":   (180, 180, 180),
    "PAPER":   (255, 220,   0),
    "PLASTIC": (0,   150, 255),
    "SHORE":   (255, 140,   0),
}

# ── Shared state ───────────────────────────────────────────────
state = {
    "detections": 0,
    "fps": 0,
    "total_collected": 0,
    "model_version": 0,
    "is_training": False,
    "camera_on": False,
    "detection_log": [],   # last N detections
}
state_lock   = threading.Lock()
frame_q      = queue.Queue(maxsize=1)
result_q     = queue.Queue(maxsize=1)
stop_evt     = threading.Event()
cap_ref      = {"cap": None}

# ── Load models ────────────────────────────────────────────────
general     = None
waste_models = {}
model_lock   = threading.Lock()

def load_models():
    global general, waste_models
    if not YOLO_AVAILABLE:
        return
    try:
        general = YOLO("yolov8n.pt")
        model_files = {
            "METAL":   BASE / "metal_detector/weights/best.pt",
            "PAPER":   BASE / "paper_detector/weights/best.pt",
            "PLASTIC": BASE / "plastic_detector/weights/best.pt",
            "SHORE":   BASE / "shore_waste_detector/weights/best.pt",
        }
        for key, path in model_files.items():
            if path.exists():
                waste_models[key] = YOLO(str(path))
        print(f"✅ Models loaded: general + {list(waste_models.keys())}")
    except Exception as e:
        print(f"⚠️  Model load error: {e}")

# ── Drawing helper ─────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, label, color, conf=None):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}" if conf is not None else label
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_text = max(y1 - 1, th + 8)
    cv2.rectangle(frame, (x1, y_text - th - 8), (x1 + tw + 8, y_text), color, -1)
    brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
    txt_color  = (0, 0, 0) if brightness > 160 else (255, 255, 255)
    cv2.putText(frame, text, (x1 + 4, y_text - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)

# ── Capture thread ─────────────────────────────────────────────
def capture_thread(cap):
    frame_count = 0
    while not stop_evt.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue
        if frame_q.full():
            try: frame_q.get_nowait()
            except queue.Empty: pass
        frame_q.put(frame)

# ── Inference thread ───────────────────────────────────────────
def inference_thread(orig_w, orig_h):
    fps_time  = time.time()
    fps_count = 0
    fps_disp  = 0

    while not stop_evt.is_set():
        try:
            frame = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue

        count        = 0
        detected_now = []

        if general is not None:
            scale  = INFERENCE_WIDTH / orig_w
            inf_h  = int(orig_h * scale)
            small  = cv2.resize(frame, (INFERENCE_WIDTH, inf_h))

            results = general(small, conf=0.25, verbose=False)[0]
            for box in results.boxes:
                sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])
                conf     = float(box.conf[0])
                cls_name = results.names[int(box.cls[0])]
                label, color = LABELS.get(cls_name, (f"UNKNOWN - {cls_name}", (80,80,80)))

                x1 = int(sx1 / scale); y1 = int(sy1 / scale)
                x2 = int(sx2 / scale); y2 = int(sy2 / scale)
                draw_box(frame, x1, y1, x2, y2, label, color, conf)
                count += 1
                detected_now.append({"label": label, "conf": round(conf, 2)})

        # FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps_disp  = fps_count
            fps_count = 0
            fps_time  = time.time()

        # Update shared state
        with state_lock:
            state["detections"] = count
            state["fps"]        = fps_disp
            state["camera_on"]  = True
            if detected_now:
                state["detection_log"] = (detected_now + state["detection_log"])[:50]

        # Overlay minimal HUD
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (10, 10, 10), -1)
        cv2.putText(frame,
            f"PROJECT ALPHA  |  {count} detected  |  {fps_disp} fps",
            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if result_q.full():
            try: result_q.get_nowait()
            except queue.Empty: pass
        result_q.put(frame)

# ── MJPEG generator ────────────────────────────────────────────
def mjpeg_generator():
    while not stop_evt.is_set():
        try:
            frame = result_q.get(timeout=1.0)
        except queue.Empty:
            # Send a blank frame so the stream doesn't hang
            blank = 255 * __import__('numpy').ones((480, 640, 3), dtype='uint8')
            _, buf = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
            continue

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")

# ── Routes ─────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html") as f:
        return f.read()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/stats")
async def stats():
    with state_lock:
        return dict(state)

@app.post("/api/camera/start")
async def camera_start():
    global cap_ref
    if cap_ref["cap"] is not None and cap_ref["cap"].isOpened():
        return {"ok": True, "msg": "Already running"}

    stop_evt.clear()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"ok": False, "msg": "Cannot open camera"}

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ret, f0 = cap.read()
    if not ret:
        return {"ok": False, "msg": "Could not read first frame"}

    cap_ref["cap"] = cap
    orig_h, orig_w = f0.shape[:2]

    threading.Thread(target=capture_thread,   args=(cap,),           daemon=True).start()
    threading.Thread(target=inference_thread, args=(orig_w, orig_h), daemon=True).start()

    with state_lock:
        state["camera_on"] = True
    return {"ok": True, "msg": "Camera started"}

@app.post("/api/camera/stop")
async def camera_stop():
    stop_evt.set()
    if cap_ref["cap"]:
        cap_ref["cap"].release()
        cap_ref["cap"] = None
    with state_lock:
        state["camera_on"] = False
    return {"ok": True, "msg": "Camera stopped"}

# ── Startup ────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    load_models()

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
