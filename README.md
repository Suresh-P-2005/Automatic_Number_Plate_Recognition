# 🚗 Automatic Number Plate Recognition (ANPR) System

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-Supported-FF6F00?style=for-the-badge)](https://github.com/JaidedAI/EasyOCR)

An end-to-end computer vision and deep learning web application for **Automatic Number Plate Recognition (ANPR)**. Built with **FastAPI**, **YOLOv11** custom models, **OpenCV**, and **EasyOCR**, this system detects vehicles, localizes license plates, performs OCR text extraction, validates plate syntax, and tracks results across live webcam feeds, IP RTSP streams, uploaded images, and recorded videos.

---

## ✨ Key Features

- 🖼️ **Image ANPR Processing**: Upload static images (JPG, PNG) to detect vehicles and license plates with bounding box visualizer, OCR text extraction, confidence metrics, and cropped plate displays.
- 🎥 **Video Stream Analytics**: Upload MP4/AVI videos for frame-by-frame detection, multi-frame confidence voting, automated `ffmpeg` web-optimized video rendering, and full detection playback.
- 📷 **Real-Time Browser Webcam**: Interactive live webcam interface with asynchronous frame processing, continuous detection, and live log updates.
- 🌐 **IP Camera Stream**: RTSP / HTTP video stream integration for real-time traffic, security, and parking surveillance monitoring.
- 🎯 **Dual YOLOv11 Model Pipeline**: 
  - **Vehicle Detector** (`vehicle_best.pt`): Detects cars, buses, trucks, and motorcycles.
  - **License Plate Localization** (`plate_best.pt`): Fine-tuned model for high-precision plate bounding box extraction.
- 🔬 **Advanced Image Preprocessing**: 4x super-resolution cubic scaling, bilateral filtering, grayscale conversion, and adaptive thresholding to maximize OCR recognition accuracy.
- 🛡️ **Regex Syntax Validation & Correction**: Built-in pattern recognition (e.g. Indian standard formats like `TN38AB1234`) with automatic character correction (resolving `0` vs `O`, `1` vs `I`, etc.).
- 📊 **CSV Export & Audit Logging**: Automatically exports recognized license plates, frame indices, and detection timestamps to structured CSV files.

---

## 🏗️ System Architecture & Workflow

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│   Input Source  │     │   Vehicle Detection  │     │  License Plate Local.  │
│ (Image/Video/   │ ──► │  (YOLOv11 Custom)    │ ──► │   (YOLOv11 Custom)     │
│  Webcam/RTSP)   │     └──────────────────────┘     └───────────┬────────────┘
└─────────────────┘                                              │
                                                                 ▼
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│ Web Dashboard   │     │  Format Validation   │     │  Image Preprocessing   │
│   & CSV Export  │ ◄── │  & Regex Correction  │ ◄── │   & OCR Text Extraction│
│                 │     │                      │     │  (OpenCV + EasyOCR)    │
└─────────────────┘     └──────────────────────┘     └───────────┬────────────┘
```

---

## 📁 Repository Structure

```
Automatic_Number_Plate_Recognition/
│
├── app/                        # FastAPI Web Application
│   ├── main.py                 # Application routes, websocket/endpoints & setup
│   ├── static/                 # Frontend assets (CSS, JS)
│   │   ├── style.css           # Styling rules
│   │   ├── webcam.js           # Live webcam stream logic
│   │   └── ip_camera.js        # IP camera stream handler
│   └── templates/              # HTML templates (Jinja2)
│       ├── index.html          # Main navigation dashboard
│       ├── image_result.html   # Image detection output view
│       ├── video_result.html   # Video playback & results
│       ├── webcam.html         # Live webcam stream page
│       └── ip_camera.html      # IP camera stream page
│
├── src/                        # Core AI & ANPR Pipeline
│   ├── anpr_processor.py       # Main ANPR session engine & model loader
│   ├── video_processor.py      # Video frame detection & ffmpeg encoder
│   ├── final_anpr.py           # Standalone video ANPR runner script
│   ├── ocr_utils.py            # Text cleaning & character correction
│   ├── plate_validator.py      # License plate format validation rules
│   ├── test_image_processor.py # Image test utility script
│   └── test_video_processor.py # Video test utility script
│
├── models/                     # Deep Learning Weights
│   ├── vehicle_best.pt         # Fine-tuned YOLOv11 vehicle detection model
│   └── plate_best.pt           # Fine-tuned YOLOv11 license plate detector model
│
├── test_images/                # Sample input images for testing
├── videos/                     # Sample input videos
├── uploads/                    # Storage for user-uploaded media (git-ignored)
├── outputs/                    # Processed images, videos & CSV reports (git-ignored)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclude rules
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: `3.9` or higher
- **FFmpeg**: Required for encoding video outputs for web playback.
  - *macOS*: `brew install ffmpeg`
  - *Ubuntu/Debian*: `sudo apt install ffmpeg`
  - *Windows*: Download from [FFmpeg official website](https://ffmpeg.org/download.html) and add to system PATH.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Suresh-P-2005/Automatic_Number_Plate_Recognition.git
   cd Automatic_Number_Plate_Recognition
   ```

2. **Create and activate a Virtual Environment**:
   ```bash
   # On macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🖥️ Running the Web Application

Start the FastAPI development server using `uvicorn`:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Once running, open your web browser and navigate to:
👉 **`http://localhost:8000`**

---

## 🛠️ Usage Modes

1. **Image Recognition**:
   - Select **Process Image** from the dashboard.
   - Upload an image file (`.jpg`, `.jpeg`, `.png`).
   - View detected vehicles, cropped plates, extracted text, confidence score, and validation status.

2. **Video Recognition**:
   - Select **Process Video** from the dashboard.
   - Upload a sample video file (`.mp4`, `.avi`).
   - The system processes frames, draws bounding boxes, exports a downloadable CSV log, and displays the processed video.

3. **Live Webcam**:
   - Select **Webcam ANPR**.
   - Grant camera permissions in your browser.
   - The system will continuously process frames in real-time and display recognized plate details live in the table.

4. **IP Camera Stream**:
   - Select **IP Camera ANPR**.
   - Enter your IP Camera RTSP/HTTP URL (e.g. `rtsp://username:password@ip_address:554/stream`).
   - Monitor real-time feed detections automatically.

---

## ⚙️ Configuration & Parameters

Detection thresholds and preprocessing parameters can be adjusted in [`src/anpr_processor.py`](src/anpr_processor.py):

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `VEHICLE_CONFIDENCE` | `0.40` | Minimum confidence score for vehicle detection |
| `PLATE_CONFIDENCE` | `0.45` | Minimum confidence score for license plate localization |
| `MIN_PLATE_WIDTH` | `40` | Minimum plate bounding box width in pixels |
| `MIN_PLATE_HEIGHT` | `10` | Minimum plate bounding box height in pixels |
| `MIN_ASPECT_RATIO` | `1.5` | Minimum aspect ratio (width / height) for valid plates |
| `MAX_ASPECT_RATIO` | `8.0` | Maximum aspect ratio for valid plates |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Suresh-P-2005/Automatic_Number_Plate_Recognition/issues).

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
