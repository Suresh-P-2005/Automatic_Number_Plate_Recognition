# 🚗 Automatic Number Plate Recognition using YOLO & OCR

A modern Automatic Number Plate Recognition (ANPR) system built using **Python, FastAPI, YOLO, OpenCV, and OCR**. The application detects vehicle license plates from images or live webcam streams, extracts the plate text, stores detection records, and provides a responsive web dashboard for monitoring results.

---

## 📌 Overview

This project automates vehicle license plate recognition by combining object detection and Optical Character Recognition (OCR). It provides a web-based interface for uploading images or using a live camera feed to detect and recognize license plates in real time.

The system is designed with a modular architecture, making it easy to extend with new detection models, OCR engines, or database backends.

---

## ✨ Features

- 🚗 Automatic license plate detection
- 🔍 OCR-based text recognition
- 📷 Live webcam support
- 🖼 Image upload interface
- ⚡ FastAPI REST API
- 📊 Interactive web dashboard
- 💾 Detection history storage
- 🔄 Real-time processing
- 📱 Responsive user interface
- 🐳 Docker support

---

## 🛠 Tech Stack

### Backend

- Python
- FastAPI
- OpenCV
- SQLModel
- SQLite
- Uvicorn

### AI & Computer Vision

- YOLO
- OCR
- NumPy

### Frontend

- HTML
- CSS
- JavaScript

### Development

- Docker
- Git
- GitHub

---

## 📂 Project Structure

```
Automatic_Number_Plate_Recognition_YOLO_OCR
│
├── src/
│   ├── api/
│   ├── detector/
│   ├── ocr/
│   ├── pipeline/
│   ├── storage/
│   └── tracker/
│
├── tests/
├── scripts/
├── docs/
├── static/
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Suresh-P-2005/Automatic_Number_Plate_Recognition_YOLO_OCR.git
```

Go into the project

```bash
cd Automatic_Number_Plate_Recognition_YOLO_OCR
```

Install dependencies

```bash
uv sync
```

Run the application

```bash
uv run anpr serve
```

Open your browser

```
http://localhost:8000
```

---

## 🚀 Workflow

```
Camera / Image
        │
        ▼
YOLO Plate Detection
        │
        ▼
Plate Cropping
        │
        ▼
OCR Recognition
        │
        ▼
Text Validation
        │
        ▼
Database Storage
        │
        ▼
Dashboard Display
```

---

## 📷 Application

The application supports:

- Image Upload
- Live Webcam Detection
- Automatic Plate Recognition
- Detection History
- REST API Access

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|-----------|----------|-------------|
| / | GET | Dashboard |
| /health | GET | Health Check |
| /version | GET | Version |
| /api/v1/infer | POST | Image Detection |
| /api/v1/detections | GET | Detection History |
| /ws/stream | WebSocket | Live Detection |

---

## 🎯 Future Improvements

- Vehicle Make & Model Recognition
- Multi-Camera Support
- User Authentication
- Vehicle Analytics Dashboard
- Cloud Deployment
- Email Notifications
- Mobile Application

---

## 📸 Screenshots

Add screenshots of:

- Dashboard
- Live Detection
- Detection Results
- History Page

---

## 👨‍💻 Author

**Suresh P**

GitHub:
https://github.com/Suresh-P-2005

---

## ⭐ Repository

If you found this project useful, consider giving it a star.
