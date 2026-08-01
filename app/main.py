from pathlib import Path
import sys
import shutil
import uuid
import base64
import threading
import numpy as np
import cv2

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    HTTPException
)

from fastapi.staticfiles import (
    StaticFiles
)

from fastapi.templating import (
    Jinja2Templates
)


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

SRC_DIR = (
    BASE_DIR
    / "src"
)

APP_DIR = (
    BASE_DIR
    / "app"
)


# ==========================================
# ADD src TO PYTHON PATH
# ==========================================

if str(SRC_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_DIR)
    )


# ==========================================
# IMPORT ANPR MODULES
# ==========================================

from anpr_processor import (
    process_image,
    ANPRSession
)

from video_processor import (
    process_video
)


# ==========================================
# WEBCAM ANPR SESSION
# ==========================================

webcam_anpr = ANPRSession()

# ==========================================
# IP CAMERA VARIABLES
# ==========================================

ip_camera_capture = None

ip_camera_anpr = ANPRSession()

ip_camera_url = None

ip_camera_lock = threading.Lock()
# ==========================================
# ENCODE FRAME TO BASE64
# ==========================================

def frame_to_base64(
    frame
):

    success, encoded_frame = (
        cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                70
            ]
        )
    )


    if not success:

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not encode "
                "camera frame."
            )
        )


    return (
        base64.b64encode(
            encoded_frame.tobytes()
        )
        .decode(
            "utf-8"
        )
    )

# ==========================================
# PROJECT PATHS
# ==========================================

UPLOAD_IMAGE_DIR = (
    BASE_DIR
    / "uploads"
    / "images"
)

OUTPUT_IMAGE_DIR = (
    BASE_DIR
    / "outputs"
    / "images"
)

UPLOAD_VIDEO_DIR = (
    BASE_DIR
    / "uploads"
    / "videos"
)

OUTPUT_VIDEO_DIR = (
    BASE_DIR
    / "outputs"
    / "videos"
)

OUTPUT_CSV_DIR = (
    BASE_DIR
    / "outputs"
    / "csv"
)

UPLOAD_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

UPLOAD_VIDEO_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_VIDEO_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CSV_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# CREATE FASTAPI APP
# ==========================================

app = FastAPI(
    title="ANPR System",
    description=(
        "Automatic Number Plate "
        "Recognition System"
    ),
    version="1.0.0"
)


# ==========================================
# STATIC FILES
# ==========================================

app.mount(
    "/static",
    StaticFiles(
        directory=str(
            APP_DIR
            / "static"
        )
    ),
    name="static"
)

app.mount(
    "/outputs",
    StaticFiles(
        directory=str(
            BASE_DIR
            / "outputs"
        )
    ),
    name="outputs"
)

# ==========================================
# HTML TEMPLATES
# ==========================================

templates = Jinja2Templates(
    directory=str(
        APP_DIR
        / "templates"
    )
)


# ==========================================
# HOME PAGE
# ==========================================

@app.get(
    "/"
)

def home(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

# ==========================================
# PROCESS IMAGE
# ==========================================

@app.post(
    "/process-image"
)

async def process_uploaded_image(
    request: Request,
    image: UploadFile = File(...)
):

    # Check uploaded file type

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    file_extension = (
        Path(
            image.filename
        )
        .suffix
        .lower()
    )


    if (
        file_extension
        not in allowed_extensions
    ):

        return {
            "error": (
                "Only JPG, JPEG and PNG "
                "images are allowed."
            )
        }


    # Create unique file names

    unique_id = str(
        uuid.uuid4()
    )


    input_filename = (
        f"{unique_id}"
        f"{file_extension}"
    )


    output_filename = (
        f"result_"
        f"{unique_id}"
        f".jpg"
    )


    input_path = (
        UPLOAD_IMAGE_DIR
        / input_filename
    )


    output_path = (
        OUTPUT_IMAGE_DIR
        / output_filename
    )


    # Save uploaded image

    with open(
        input_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            image.file,
            buffer
        )


    # Read image using OpenCV

    input_image = cv2.imread(
        str(input_path)
    )


    if input_image is None:

        return {
            "error": (
                "The uploaded image "
                "could not be read."
            )
        }


    # Run ANPR

    processed_image, results = (
        process_image(
            input_image
        )
    )


    # Save processed image

    cv2.imwrite(
        str(output_path),
        processed_image
    )


    # Return result page

    return templates.TemplateResponse(
        request=request,
        name="image_result.html",
        context={
            "results": results,
            "output_image": (
                f"/outputs/images/"
                f"{output_filename}"
            )
        }
    )

# ==========================================
# PROCESS VIDEO
# ==========================================

@app.post(
    "/process-video"
)

async def process_uploaded_video(
    request: Request,
    video: UploadFile = File(...)
):

    # Allowed video formats

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    }

    file_extension = (
        Path(
            video.filename
        )
        .suffix
        .lower()
    )

    if (
        file_extension
        not in allowed_extensions
    ):

        return {
            "error": (
                "Only MP4, AVI, MOV and "
                "MKV videos are allowed."
            )
        }


    # Create unique file names

    unique_id = str(
        uuid.uuid4()
    )

    input_filename = (
        f"{unique_id}"
        f"{file_extension}"
    )

    output_filename = (
        f"processed_"
        f"{unique_id}"
        f".mp4"
    )

    csv_filename = (
        f"results_"
        f"{unique_id}"
        f".csv"
    )


    # Create complete paths

    input_path = (
        UPLOAD_VIDEO_DIR
        / input_filename
    )

    output_path = (
        OUTPUT_VIDEO_DIR
        / output_filename
    )

    csv_path = (
        OUTPUT_CSV_DIR
        / csv_filename
    )


    # Save uploaded video

    with open(
        input_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            video.file,
            buffer
        )


    # Run ANPR video processing

    result = process_video(
        input_video_path=input_path,
        output_video_path=output_path,
        csv_output_path=csv_path
    )


    # Return the video result page

    return templates.TemplateResponse(
        request=request,
        name="video_result.html",
        context={
            "results": (
                result["results"]
            ),

            "frames_processed": (
                result[
                    "frames_processed"
                ]
            ),

            "total_vehicles": (
                result[
                    "total_recognised_vehicles"
                ]
            ),

            "output_video": (
                f"/outputs/videos/"
                f"{output_filename}"
            ),

            "csv_file": (
                f"/outputs/csv/"
                f"{csv_filename}"
            )
        }
    )

# ==========================================
# WEBCAM PAGE
# ==========================================

@app.get(
    "/webcam"
)

def webcam_page(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="webcam.html"
    )

# ==========================================
# PROCESS WEBCAM FRAME
# ==========================================

@app.post(
    "/process-webcam-frame"
)

async def process_webcam_frame(
    frame: UploadFile = File(...)
):

    # Read uploaded frame

    frame_bytes = (
        await frame.read()
    )


    # Convert bytes to NumPy

    frame_array = (
        np.frombuffer(
            frame_bytes,
            dtype=np.uint8
        )
    )


    # Decode JPEG

    image = cv2.imdecode(
        frame_array,
        cv2.IMREAD_COLOR
    )


    if image is None:

        return {
            "status": (
                "Invalid webcam frame."
            ),

            "image": "",

            "results": []
        }


    # Process using persistent session

    processed_frame, results = (
        webcam_anpr.process_frame(
            image
        )
    )


    # Encode processed frame

    success, encoded_image = (
        cv2.imencode(
            ".jpg",
            processed_frame
        )
    )


    if not success:

        return {
            "status": (
                "Could not encode "
                "processed frame."
            ),

            "image": "",

            "results": []
        }


    # Convert image to Base64

    image_base64 = (
        base64.b64encode(
            encoded_image.tobytes()
        )
        .decode(
            "utf-8"
        )
    )


    # Create status

    if results:

        status = (
            f"{len(results)} "
            "vehicle(s) detected."
        )

    else:

        status = (
            "No vehicle detected."
        )


    return {
        "status": status,

        "image": image_base64,

        "results": results
    }

# ==========================================
# IP CAMERA PAGE
# ==========================================

@app.get(
    "/ip-camera"
)

def ip_camera_page(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="ip_camera.html"
    )

# ==========================================
# CONNECT IP CAMERA
# ==========================================

@app.post(
    "/connect-ip-camera"
)

async def connect_ip_camera(
    request: Request
):

    global ip_camera_capture
    global ip_camera_url
    global ip_camera_anpr


    data = (
        await request.json()
    )


    camera_url = (
        data.get(
            "camera_url",
            ""
        )
        .strip()
    )


    if not camera_url:

        raise HTTPException(
            status_code=400,
            detail=(
                "Camera URL is required."
            )
        )


    # Close old camera connection

    if (
        ip_camera_capture
        is not None
    ):

        ip_camera_capture.release()

        ip_camera_capture = None


    # Open new camera stream

    camera = (
        cv2.VideoCapture(
            camera_url
        )
    )

    # Keep only the newest frame.
    # Prevent old frames from building up.

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    if (
        not camera.isOpened()
    ):

        camera.release()


        raise HTTPException(
            status_code=400,
            detail=(
                "Could not connect to "
                "the IP camera. "
                "Check the URL and "
                "make sure the camera "
                "is on the same network."
            )
        )


    # Read one frame to verify stream

    success, frame = (
        camera.read()
    )


    if (
        not success
        or frame is None
    ):

        camera.release()


        raise HTTPException(
            status_code=400,
            detail=(
                "Camera connected, but "
                "no video frame was received."
            )
        )


    # Save camera connection

    ip_camera_capture = (
        camera
    )


    ip_camera_url = (
        camera_url
    )


    # Reset OCR session

    ip_camera_anpr = (
        ANPRSession()
    )


    return {
        "message": (
            "IP camera connected."
        )
    }

# ==========================================
# PROCESS IP CAMERA FRAME
# ==========================================

@app.post(
    "/process-ip-camera-frame"
)

def process_ip_camera_frame():

    global ip_camera_capture


    if (
        ip_camera_capture
        is None
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "No IP camera is connected."
            )
        )


    # Read the latest camera frame

    with ip_camera_lock:

        success, frame = (
            ip_camera_capture.read()
        )


    if (
        not success
        or frame is None
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not read a frame "
                "from the IP camera."
            )
        )


    # ==========================================
    # RESIZE FRAME FOR FASTER ANPR
    # ==========================================

    frame_height, frame_width = (
        frame.shape[:2]
    )


    MAX_FRAME_WIDTH = 960


    if (
        frame_width
        > MAX_FRAME_WIDTH
    ):

        scale = (
            MAX_FRAME_WIDTH
            / frame_width
        )


        new_height = int(
            frame_height
            * scale
        )


        frame = cv2.resize(
            frame,
            (
                MAX_FRAME_WIDTH,
                new_height
            ),
            interpolation=
                cv2.INTER_AREA
        )


    # Keep original frame

    original_frame = (
        frame.copy()
    )


    # Process ANPR

    processed_frame, results = (
        ip_camera_anpr.process_frame(
            frame
        )
    )


    # Convert frames to Base64

    original_image = (
        frame_to_base64(
            original_frame
        )
    )


    processed_image = (
        frame_to_base64(
            processed_frame
        )
    )


    # Create status

    if results:

        status = (
            f"{len(results)} "
            "vehicle(s) detected."
        )

    else:

        status = (
            "No vehicle detected."
        )


    return {
        "status": status,

        "original_image":
            original_image,

        "processed_image":
            processed_image,

        "results":
            results
    }

# ==========================================
# GET IP CAMERA LIVE FRAME
# ==========================================

@app.get(
    "/ip-camera-live-frame"
)

def get_ip_camera_live_frame():

    global ip_camera_capture


    if (
        ip_camera_capture
        is None
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "No IP camera is connected."
            )
        )


    with ip_camera_lock:
        success, frame = (
            ip_camera_capture.read()
        )


    if (
        not success
        or frame is None
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not read a live "
                "frame from the IP camera."
            )
        )


    # Resize only for browser display

    frame_height, frame_width = (
        frame.shape[:2]
    )


    LIVE_FRAME_WIDTH = 960


    if (
        frame_width
        > LIVE_FRAME_WIDTH
    ):

        scale = (
            LIVE_FRAME_WIDTH
            / frame_width
        )


        new_height = int(
            frame_height
            * scale
        )


        frame = cv2.resize(
            frame,
            (
                LIVE_FRAME_WIDTH,
                new_height
            ),
            interpolation=
                cv2.INTER_AREA
        )


    return {
        "image":
            frame_to_base64(
                frame
            )
    }

# ==========================================
# DISCONNECT IP CAMERA
# ==========================================

@app.post(
    "/disconnect-ip-camera"
)

def disconnect_ip_camera():

    global ip_camera_capture
    global ip_camera_url


    if (
        ip_camera_capture
        is not None
    ):

        ip_camera_capture.release()

        ip_camera_capture = None


    ip_camera_url = None


    return {
        "message": (
            "IP camera disconnected."
        )
    }    

# ==========================================
# HEALTH CHECK
# ==========================================

@app.get(
    "/health"
)

def health_check():

    return {
        "status": "running",
        "message": (
            "ANPR FastAPI backend "
            "is working"
        )
    }