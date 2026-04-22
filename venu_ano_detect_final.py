import csv
import os
import threading
import time
from datetime import datetime
from typing import List

from flask import Flask, Response, jsonify, send_from_directory

# Attempt to import Jetson libraries
try:
    import jetson.inference
    import jetson.utils
    HAS_JETSON = True
except ImportError:
    HAS_JETSON = False

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CAMERA_PATH = "/dev/video0"
MODEL_NAME = "ssd-mobilenet-v2"
LOG_FILE = "anomaly_log.csv"
DASHBOARD_FILE = "spiderman_dashboard_final.html"

# Detection labels expected from COCO-style models
FORBIDDEN_OBJECTS = {
    "wine glass": "wine glass",
    "bottle": "wine bottle",
    "cell phone": "phone",
    "cup": "cup",
    "remote": "remote",
    "laptop": "laptop",
}

PERSON_LIMIT = 2
CONF_THRESHOLD = 0.50
COOLDOWN_SECONDS = 3

# -----------------------------------------------------------------------------
# Shared state
# -----------------------------------------------------------------------------
state = {
    "anomalies": 0,
    "anomaly_count": 0,
    "last_anomaly": "None",
    "last_anomaly_type": "",
    "last_anomaly_reason": "",
    "last_anomaly_time": 0.0,
    "time_since_last": "N/A",
    "current_log": [],
    "frame": 0,
    "persons": 0,
    "objects": 0,
    "avg_conf": 0,
    "fps": 0,
    "status": "online",
}

state_lock = threading.Lock()
latest_frame = None
latest_frame_lock = threading.Lock()
last_logged = {}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def init_csv():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Anomaly_Type", "Details"])

def timestamp_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def cooldown_ok(key: str) -> bool:
    now = time.time()
    prev = last_logged.get(key, 0)
    if now - prev >= COOLDOWN_SECONDS:
        last_logged[key] = now
        return True
    return False

def log_anomaly(anomaly_type: str, details: str):
    ts = timestamp_str()
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, anomaly_type, details])

    with state_lock:
        state["anomalies"] += 1
        state["anomaly_count"] = state["anomalies"]
        state["last_anomaly"] = f"{anomaly_type}: {details}"
        state["last_anomaly_type"] = anomaly_type
        state["last_anomaly_reason"] = details
        state["last_anomaly_time"] = time.time()
        state["current_log"].insert(0, {
            "time": ts,
            "type": anomaly_type,
            "details": details
        })
        state["current_log"] = state["current_log"][:20]

def update_time_since_last():
    with state_lock:
        if state["last_anomaly_time"] > 0:
            diff = int(time.time() - state["last_anomaly_time"])
            state["time_since_last"] = f"{diff}s ago"
        else:
            state["time_since_last"] = "N/A"

def draw_hud_box(img, x1, y1, x2, y2, label, color=(232, 217, 5)):
    import cv2

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    w = x2 - x1
    h = y2 - y1
    r = max(int(max(w, h) * 0.55), 20)

    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
    cv2.line(img, (cx - 15, cy), (cx - 5, cy), color, 1)
    cv2.line(img, (cx + 5, cy), (cx + 15, cy), color, 1)
    cv2.line(img, (cx, cy - 15), (cx, cy - 5), color, 1)
    cv2.line(img, (cx, cy + 5), (cx, cy + 15), color, 1)

    for a1, a2 in [(15, 75), (105, 165), (195, 255), (285, 345)]:
        cv2.ellipse(img, (cx, cy), (r, r), 0, a1, a2, color, 2)

    line_len = max(int(min(w, h) * 0.2), 10)
    cv2.line(img, (x1, y1), (x1 + line_len, y1), color, 2)
    cv2.line(img, (x1, y1), (x1, y1 + line_len), color, 2)
    cv2.line(img, (x2, y1), (x2 - line_len, y1), color, 2)
    cv2.line(img, (x2, y1), (x2, y1 + line_len), color, 2)
    cv2.line(img, (x1, y2), (x1 + line_len, y2), color, 2)
    cv2.line(img, (x1, y2), (x1, y2 - line_len), color, 2)
    cv2.line(img, (x2, y2), (x2 - line_len, y2), color, 2)
    cv2.line(img, (x2, y2), (x2, y2 - line_len), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, 0.5, 1)[0]
    tx = cx - int(text_size[0] / 2)
    ty = cy + r + 15
    cv2.rectangle(img, (tx - 2, ty - text_size[1] - 4), (tx + text_size[0] + 2, ty + 2), (0, 0, 0), -1)
    cv2.putText(img, label, (tx, ty - 2), font, 0.5, color, 1)

def analyze_detections(labels: List[str]) -> List[tuple]:
    anomalies = []

    person_count = sum(1 for x in labels if x == "person")
    forbidden_found = [FORBIDDEN_OBJECTS[x] for x in labels if x in FORBIDDEN_OBJECTS]

    if forbidden_found:
        unique_items = sorted(set(forbidden_found))
        key = "forbidden_" + "_".join(unique_items)
        if cooldown_ok(key):
            anomalies.append(("FORBIDDEN", f"Detected forbidden object(s): {', '.join(unique_items)}"))

    if person_count > PERSON_LIMIT:
        key = "count"
        if cooldown_ok(key):
            anomalies.append(("COUNT", f"Detected {person_count} people (> {PERSON_LIMIT})"))

    return anomalies

# -----------------------------------------------------------------------------
# Detection loop
# -----------------------------------------------------------------------------
def detection_loop():
    import cv2
    import numpy as np

    global latest_frame

    prev_time = time.time()

    if HAS_JETSON:
        try:
            net = jetson.inference.detectNet(MODEL_NAME, threshold=CONF_THRESHOLD)
            camera = jetson.utils.videoSource(CAMERA_PATH)
            print("Jetson backend started.")
        except Exception as e:
            print(f"Jetson init failed: {e}")
            return

        while True:
            img = camera.Capture()
            if img is None:
                time.sleep(0.01)
                continue

            detections = net.Detect(img, overlay="none")
            img_np = jetson.utils.cudaToNumpy(img)
            if img_np.dtype == np.float32:
                img_np = img_np.astype(np.uint8)

            if img_np.shape[2] == 4:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            labels = []
            total_conf = 0.0
            det_count = 0

            for d in detections:
                class_name = net.GetClassDesc(d.ClassID).lower()
                conf = float(d.Confidence)
                labels.append(class_name)
                total_conf += conf
                det_count += 1

                x1, y1, x2, y2 = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
                color = (0, 0, 255) if class_name in FORBIDDEN_OBJECTS else (232, 217, 5)
                pretty = FORBIDDEN_OBJECTS.get(class_name, class_name)
                label = f"{pretty.upper()} {conf:.2f}"
                draw_hud_box(img_bgr, x1, y1, x2, y2, label, color)

            anomalies = analyze_detections(labels)
            for a_type, reason in anomalies:
                log_anomaly(a_type, reason)

            person_count = sum(1 for x in labels if x == "person")
            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            if anomalies:
                cv2.putText(img_bgr, "THREAT DETECTED", (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            ok, buffer = cv2.imencode(".jpg", img_bgr)
            if ok:
                with latest_frame_lock:
                    latest_frame = buffer.tobytes()

            with state_lock:
                state["frame"] += 1
                state["persons"] = person_count
                state["objects"] = det_count
                state["avg_conf"] = int((total_conf / det_count) * 100) if det_count else 0
                state["fps"] = int(fps)
            update_time_since_last()

    else:
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            print("OpenCV/YOLO fallback backend started.")
        except Exception as e:
            print(f"Fallback init failed: {e}")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam.")
            return

        while True:
            success, img_bgr = cap.read()
            if not success:
                time.sleep(0.01)
                continue

            results = model(img_bgr, verbose=False)
            labels = []
            total_conf = 0.0
            det_count = 0

            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = str(model.names[class_id]).lower()
                conf = float(box.conf[0])
                labels.append(class_name)
                total_conf += conf
                det_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 0, 255) if class_name in FORBIDDEN_OBJECTS else (232, 217, 5)
                pretty = FORBIDDEN_OBJECTS.get(class_name, class_name)
                label = f"{pretty.upper()} {conf:.2f}"
                draw_hud_box(img_bgr, x1, y1, x2, y2, label, color)

            anomalies = analyze_detections(labels)
            for a_type, reason in anomalies:
                log_anomaly(a_type, reason)

            person_count = sum(1 for x in labels if x == "person")
            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            if anomalies:
                cv2.putText(img_bgr, "THREAT DETECTED", (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            ok, buffer = cv2.imencode(".jpg", img_bgr)
            if ok:
                with latest_frame_lock:
                    latest_frame = buffer.tobytes()

            with state_lock:
                state["frame"] += 1
                state["persons"] = person_count
                state["objects"] = det_count
                state["avg_conf"] = int((total_conf / det_count) * 100) if det_count else 0
                state["fps"] = int(fps)
            update_time_since_last()

# -----------------------------------------------------------------------------
# Flask routes
# -----------------------------------------------------------------------------
@app.route("/")
def root():
    return send_from_directory(os.getcwd(), DASHBOARD_FILE)

@app.route("/status")
def status():
    with state_lock:
        return jsonify(state)

@app.route("/stream")
def stream():
    def generate():
        while True:
            with latest_frame_lock:
                frame = latest_frame
            if frame is None:
                time.sleep(0.03)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/logs")
def logs():
    if not os.path.exists(LOG_FILE):
        init_csv()
    return send_from_directory(os.getcwd(), LOG_FILE, as_attachment=True)

if __name__ == "__main__":
    init_csv()
    worker = threading.Thread(target=detection_loop, daemon=True)
    worker.start()
    print("Starting W.E.B Spiderman Threat Detection Server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
