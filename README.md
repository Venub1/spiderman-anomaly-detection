# 🕷️ W.E.B – Spiderman Anomaly Detection System

A real-time anomaly detection system deployed on NVIDIA Jetson, combining computer vision, rule-based intelligence, and an interactive Spiderman-themed dashboard for live monitoring and alerting.

---

## 🧠 Project Overview

This project is designed to simulate a **real-world intelligent monitoring system** where:

* A live camera feed is analyzed using deep learning
* Detected objects are evaluated against predefined **anomaly rules**
* Violations trigger **real-time alerts**
* All events are logged and visualized in a **cyberpunk-style dashboard**

---

## 🎯 Objectives

* Build a **real-time edge AI system**
* Detect **context-based anomalies**, not just objects
* Create an **interactive monitoring dashboard**
* Enable **alert-driven decision support**
* Demonstrate an **end-to-end pipeline (camera → model → UI)**

---

## ⚙️ System Architecture

```
Camera Input (Jetson)
        ↓
Object Detection Model (SSD-Mobilenet-v2)
        ↓
Rule Engine (Anomaly Detection Logic)
        ↓
Flask Backend (API + Stream)
        ↓
Spiderman Dashboard (Frontend UI)
```

---

## 🧠 Model & Detection

### Model Used

* `SSD-Mobilenet-v2` (Jetson Inference)
* Optimized for **real-time edge deployment**

### Detection Capabilities

* Object classification
* Bounding box localization
* Confidence scoring

---

## 🚨 Anomaly Detection Logic

This system moves beyond basic detection and applies **context-aware rules**.

### 1. Forbidden Object Detection

Triggers when any of the following are detected:

* 📱 Phone (`cell phone`)
* 🍷 Wine glass
* 🍾 Wine bottle (`bottle`)
* ☕ Cup
* 📺 Remote
* 💻 Laptop

---

### 2. Person Count Violation

* If number of detected people > 2
  → 🚨 **Alert triggered**

---

## ⚡ Backend System (Flask)

The backend serves as the **central controller** of the system.

### Endpoints

| Endpoint  | Description            |
| --------- | ---------------------- |
| `/`       | Loads dashboard        |
| `/stream` | Live video feed        |
| `/status` | Real-time system stats |
| `/logs`   | Download anomaly CSV   |

---

### Key Responsibilities

* Run detection loop
* Apply anomaly rules
* Maintain system state
* Log anomalies to CSV
* Serve real-time data to UI

---

## 📊 Dashboard (W.E.B UI)

A custom-built **Spiderman-themed cyberpunk interface**.

### Features

* 🎥 Live video feed
* 📈 Real-time stats (FPS, confidence, objects)
* 🚨 Popup alerts on anomaly detection
* 📜 Threat log panel
* 🧠 Spider-sense visualization
* 📊 Threat level indicator

---

### Alert System

When anomaly occurs:

* Red UI mode activates
* Popup appears showing:

  * Type of anomaly
  * Reason
  * Timestamp
* Dashboard updates instantly

---

## 📁 Logging System

All anomalies are stored in:

```
anomaly_log.csv
```

### Example:

```csv
Timestamp,Anomaly_Type,Details
2026-04-22 15:12:01,FORBIDDEN,Detected laptop
2026-04-22 15:12:05,COUNT,Detected 4 people (>2)
```

---

## 🚀 How to Run

```bash
python3 venu_ano_detect_final.py
```

Then open:

```
http://localhost:5000
```

Or from another device:

```
http://<jetson-ip>:5000
```

---

## 🛠 Tech Stack

* Python
* Flask
* OpenCV
* NVIDIA Jetson Inference
* HTML / CSS / JavaScript

---

## 🧠 Design Philosophy

> **“Detection is not intelligence — interpretation is.”**

Instead of just identifying objects, the system evaluates **what those objects mean in context**, enabling:

* smarter alerts
* reduced noise
* actionable insights

---

## ⚠️ Limitations

* Depends on COCO class labels
* Lighting conditions affect accuracy
* No object tracking (yet)
* Rule-based (not learned behavior)

---

## 🔮 Future Improvements

* 🎯 Object tracking (IDs per person)
* 🧠 Behavior-based anomaly detection
* 📊 Dashboard analytics & trends
* 📸 Save anomaly snapshots
* ⚡ TensorRT optimization
* 📱 Mobile dashboard support

---

## 📸 Demo

*(Add your screenshots here)*

```
![Dashboard](screenshots/dashboard.png)
```

---

## 👤 Author

**Venu Bandi**
MPS Data Science – UMBC

---

## ⭐ Final Note

This project demonstrates how **edge AI + real-time visualization + rule-based reasoning** can be combined to build intelligent monitoring systems.
