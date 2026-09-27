"""
Automatic Number Plate Recognition (ANPR) System
Gradio Interface for Hugging Face ZeroGPU Spaces
"""

from pathlib import Path
import sys
import uuid
import cv2
import numpy as np
import gradio as gr
import spaces

# ==================================================
# PATHS & PYTHON PATH
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR  = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OUTPUT_IMAGE_DIR = BASE_DIR / "outputs" / "images"
OUTPUT_VIDEO_DIR = BASE_DIR / "outputs" / "videos"
OUTPUT_CSV_DIR   = BASE_DIR / "outputs" / "csv"

for d in [OUTPUT_IMAGE_DIR, OUTPUT_VIDEO_DIR, OUTPUT_CSV_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==================================================
# LOAD ANPR MODULES (models load on CPU at startup)
# ==================================================

from anpr_processor import process_image, ANPRSession
from video_processor import process_video

# We initialize these lazily to avoid triggering GPU memory
# allocation during the global app startup phase. 
_webcam_session    = None
_ip_camera_session = None
_ip_camera_capture = None   # cv2.VideoCapture handle

def get_webcam_session():
    global _webcam_session
    if _webcam_session is None:
        _webcam_session = ANPRSession()
    return _webcam_session

def get_ip_camera_session():
    global _ip_camera_session
    if _ip_camera_session is None:
        _ip_camera_session = ANPRSession()
    return _ip_camera_session

# ==================================================
# HELPERS
# ==================================================

def _results_to_markdown(results: list) -> str:
    """Convert ANPR result list to a Markdown table."""
    if not results:
        return "⚠️ **No vehicles or licence plates detected.**"
    rows = ["| # | Vehicle ID | Licence Plate | Readings |",
            "|---|-----------|--------------|---------|"]
    for i, r in enumerate(results, 1):
        plate = r.get("license_plate", "—")
        vid   = r.get("vehicle_id", "—")
        reads = r.get("number_of_readings", "—")
        rows.append(f"| {i} | {vid} | **{plate}** | {reads} |")
    return "\n".join(rows)


def _bgr_to_rgb(image):
    """Convert OpenCV BGR image to RGB for Gradio."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ==================================================
# 1. IMAGE ANPR
# ==================================================

@spaces.GPU
def run_image_anpr(image_rgb):
    """
    Takes an RGB numpy array from Gradio, runs ANPR,
    returns (annotated_image_rgb, results_markdown).
    """
    if image_rgb is None:
        return None, "❌ Please upload an image first."

    # Gradio gives RGB; convert to BGR for OpenCV/YOLO
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Models inside process_image will be moved to CUDA safely here
    processed_bgr, results = process_image(image_bgr)
    processed_rgb = _bgr_to_rgb(processed_bgr)

    return processed_rgb, _results_to_markdown(results)


# ==================================================
# 2. VIDEO ANPR
# ==================================================

@spaces.GPU(duration=120)
def run_video_anpr(video_path):
    """
    Accepts a video file path from Gradio, runs ANPR on
    every frame, returns (output_video_path, results_md,
    csv_file_path).
    """
    if video_path is None:
        return None, "❌ Please upload a video first.", None

    uid          = str(uuid.uuid4())
    output_path  = OUTPUT_VIDEO_DIR / f"processed_{uid}.mp4"
    csv_path     = OUTPUT_CSV_DIR   / f"results_{uid}.csv"

    # process_video must handle its own CUDA operations inside
    result = process_video(
        input_video_path  = video_path,
        output_video_path = output_path,
        csv_output_path   = csv_path,
    )

    summary = (
        f"✅ **Processing complete!**\n\n"
        f"- Frames processed : **{result['frames_processed']}**\n"
        f"- Vehicles recognised : **{result['total_recognised_vehicles']}**\n\n"
        + _results_to_markdown(result["results"])
    )

    csv_out = str(csv_path) if csv_path.exists() else None

    return str(output_path), summary, csv_out


# ==================================================
# 3. WEBCAM ANPR  (streaming, frame-by-frame)
# ==================================================

@spaces.GPU
def run_webcam_frame(frame_rgb):
    """
    Called by Gradio's streaming webcam component on each
    captured frame. Returns an annotated RGB frame.
    """
    if frame_rgb is None:
        return frame_rgb

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Lazy load session inside the GPU decorator
    session = get_webcam_session()
    processed_bgr, _ = session.process_frame(frame_bgr)
    return _bgr_to_rgb(processed_bgr)


def reset_webcam_session():
    """Clear tracking state so a new session starts fresh."""
    global _webcam_session
    _webcam_session = ANPRSession()
    return "🔄 Webcam session reset."


# ==================================================
# 4. IP CAMERA ANPR
# ==================================================

def connect_ip_camera(camera_url: str):
    """Open an RTSP / HTTP camera stream."""
    global _ip_camera_capture

    camera_url = camera_url.strip()
    if not camera_url:
        return "❌ Please enter a valid camera URL."

    # Release any existing connection
    if _ip_camera_capture is not None:
        _ip_camera_capture.release()
        _ip_camera_capture = None

    cap = cv2.VideoCapture(camera_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        return "❌ Could not connect. Check the URL and your network."

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return "❌ Connected but could not read a frame from the stream."

    _ip_camera_capture = cap
    return "✅ IP camera connected successfully!"


@spaces.GPU
def grab_ip_camera_frame():
    """
    Read one frame from the connected IP camera, run ANPR,
    return (original_rgb, annotated_rgb, status).
    """
    global _ip_camera_capture

    if _ip_camera_capture is None:
        empty = np.zeros((360, 640, 3), dtype=np.uint8)
        return empty, empty, "❌ No IP camera connected."

    ok, frame_bgr = _ip_camera_capture.read()
    if not ok or frame_bgr is None:
        empty = np.zeros((360, 640, 3), dtype=np.uint8)
        return empty, empty, "⚠️ Could not read frame."

    # Resize for speed
    h, w = frame_bgr.shape[:2]
    if w > 960:
        scale = 960 / w
        frame_bgr = cv2.resize(
            frame_bgr, (960, int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    original_rgb = _bgr_to_rgb(frame_bgr)
    
    # Lazy load session inside the GPU decorator
    session = get_ip_camera_session()
    processed_bgr, results = session.process_frame(frame_bgr)
    
    processed_rgb = _bgr_to_rgb(processed_bgr)

    status = (
        f"🚗 {len(results)} vehicle(s) detected."
        if results else "👀 No vehicles detected."
    )
    return original_rgb, processed_rgb, status


def disconnect_ip_camera():
    """Release the IP camera stream."""
    global _ip_camera_capture
    if _ip_camera_capture is not None:
        _ip_camera_capture.release()
        _ip_camera_capture = None
    return "🔌 IP camera disconnected."


# ==================================================
# GRADIO UI
# ==================================================

CSS = """
.gradio-container { font-family: 'Inter', sans-serif; }
h1 { text-align: center; }
.tab-nav button { font-size: 15px; font-weight: 600; }
"""

with gr.Blocks(
    title="🚗 ANPR System",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=CSS,
) as demo:

    gr.Markdown(
        """
        # 🚗 Automatic Number Plate Recognition (ANPR)
        Detect and read vehicle licence plates from **images**, **videos**,
        your **webcam**, or an **IP camera stream** — powered by YOLOv11 + EasyOCR.
        """
    )

    # --------------------------------------------------
    # TAB 1 — IMAGE
    # --------------------------------------------------
    with gr.Tab("📷 Image"):
        gr.Markdown("### Upload an image to detect licence plates")
        with gr.Row():
            with gr.Column():
                img_input  = gr.Image(
                    label   = "Upload Image (JPG / PNG)",
                    type    = "numpy",
                )
                img_btn    = gr.Button("🔍 Run ANPR", variant="primary")
            with gr.Column():
                img_output = gr.Image(label="Annotated Result", type="numpy")
                img_table  = gr.Markdown(label="Detection Results")

        img_btn.click(
            fn      = run_image_anpr,
            inputs  = [img_input],
            outputs = [img_output, img_table],
        )

    # --------------------------------------------------
    # TAB 2 — VIDEO
    # --------------------------------------------------
    with gr.Tab("🎬 Video"):
        gr.Markdown(
            "### Upload a video to process every frame\n"
            "Supported formats: **MP4, AVI, MOV, MKV**"
        )
        with gr.Row():
            with gr.Column():
                vid_input  = gr.Video(label="Upload Video")
                vid_btn    = gr.Button("▶️ Process Video", variant="primary")
            with gr.Column():
                vid_output  = gr.Video(label="Processed Video")
                vid_table   = gr.Markdown(label="Detection Summary")
                csv_output  = gr.File(label="⬇️ Download CSV Results")

        vid_btn.click(
            fn      = run_video_anpr,
            inputs  = [vid_input],
            outputs = [vid_output, vid_table, csv_output],
        )

    # --------------------------------------------------
    # TAB 3 — WEBCAM
    # --------------------------------------------------
    with gr.Tab("📹 Webcam"):
        gr.Markdown(
            "### Live webcam licence plate detection\n"
            "Allow browser camera access, then click **Start**."
        )
        with gr.Row():
            webcam_in  = gr.Image(
                label     = "Webcam Input",
                sources   = ["webcam"],
                streaming = True,
                type      = "numpy",
                mirror_webcam = True,
            )
            webcam_out = gr.Image(label="ANPR Output", type="numpy")

        reset_btn    = gr.Button("🔄 Reset Session")
        reset_status = gr.Markdown()

        webcam_in.stream(
            fn      = run_webcam_frame,
            inputs  = [webcam_in],
            outputs = [webcam_out],
        )
        reset_btn.click(
            fn      = reset_webcam_session,
            outputs = [reset_status],
        )

    # --------------------------------------------------
    # TAB 4 — IP CAMERA
    # --------------------------------------------------
    with gr.Tab("📡 IP Camera"):
        gr.Markdown(
            "### Connect to an IP / RTSP camera stream\n"
            "Enter an RTSP or HTTP stream URL, connect, then grab frames."
        )
        with gr.Row():
            ip_url_input = gr.Textbox(
                label       = "Camera URL",
                placeholder = "rtsp://user:pass@192.168.1.100:554/stream",
                scale       = 4,
            )
            ip_connect_btn    = gr.Button("🔗 Connect",    variant="primary", scale=1)
            ip_disconnect_btn = gr.Button("🔌 Disconnect", variant="stop",    scale=1)

        ip_status = gr.Markdown()

        ip_connect_btn.click(
            fn      = connect_ip_camera,
            inputs  = [ip_url_input],
            outputs = [ip_status],
        )
        ip_disconnect_btn.click(
            fn      = disconnect_ip_camera,
            outputs = [ip_status],
        )

        gr.Markdown("---")
        grab_btn = gr.Button("📸 Grab & Analyse Frame", variant="primary")

        with gr.Row():
            ip_original  = gr.Image(label="Original Frame",   type="numpy")
            ip_processed = gr.Image(label="Annotated Result",  type="numpy")

        ip_frame_status = gr.Markdown()

        grab_btn.click(
            fn      = grab_ip_camera_frame,
            outputs = [ip_original, ip_processed, ip_frame_status],
        )

    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------
    gr.Markdown(
        """
        ---
        Built with 🤗 **Gradio** · YOLOv11 · EasyOCR · OpenCV
        """
    )

# ==================================================
# LAUNCH
# ==================================================

if __name__ == "__main__":
    demo.launch()
