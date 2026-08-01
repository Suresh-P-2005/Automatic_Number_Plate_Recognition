from pathlib import Path
from collections import defaultdict, Counter

import cv2
import easyocr

from ultralytics import YOLO

from ocr_utils import (
    correct_plate
)

from plate_validator import (
    is_valid_plate
)


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

VEHICLE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "vehicle_best.pt"
)

PLATE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "plate_best.pt"
)


# ==================================================
# SETTINGS
# ==================================================

VEHICLE_CONFIDENCE = 0.40

PLATE_CONFIDENCE = 0.45

MIN_PLATE_WIDTH = 40

MIN_PLATE_HEIGHT = 10

MIN_ASPECT_RATIO = 1.5

MAX_ASPECT_RATIO = 8.0


# ==================================================
# LOAD MODELS ONCE
# ==================================================

print("Loading vehicle model...")

vehicle_model = YOLO(
    str(VEHICLE_MODEL_PATH)
)

print("Loading plate model...")

plate_model = YOLO(
    str(PLATE_MODEL_PATH)
)

print("Loading EasyOCR...")

ocr_reader = easyocr.Reader(
    ["en"],
    gpu=False
)

print("ANPR models loaded.")


# ==================================================
# OCR HELPERS
# ==================================================

def enhance_plate(image):

    enlarged = cv2.resize(
        image,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(
        gray
    )


def read_plate(image):

    enhanced = enhance_plate(
        image
    )

    results = ocr_reader.readtext(
        enhanced,
        detail=1,
        paragraph=False,
        allowlist=(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
        )
    )

    if not results:

        return None

    text_parts = []

    for _, text, confidence in results:

        if confidence >= 0.30:

            text_parts.append(
                text
            )

    if not text_parts:

        return None

    raw_text = "".join(
        text_parts
    )

    corrected = correct_plate(
        raw_text
    )

    if is_valid_plate(
        corrected
    ):

        return corrected

    return None


# ==================================================
# ANPR SESSION
# ==================================================

class ANPRSession:

    def __init__(self):

        # Stores valid OCR results for each vehicle ID
        self.plate_readings = defaultdict(
            list
        )

        # Counts processed frames
        self.frame_number = 0

        # Run OCR once every 5 frames
        self.ocr_interval = 5


    # ==============================================
    # GET MOST COMMON PLATE
    # ==============================================

    def get_best_plate(
        self,
        track_id
    ):

        readings = (
            self.plate_readings[
                track_id
            ]
        )

        if not readings:

            return None

        counts = Counter(
            readings
        )

        best_plate, count = (
            counts.most_common(1)[0]
        )

        # Require the same plate to appear
        # at least two times
        if count >= 2:

            return best_plate

        return None


    # ==============================================
    # PROCESS ONE FRAME
    # ==============================================

    def process_frame(
        self,
        frame
    ):

        self.frame_number += 1

        output_frame = frame.copy()

        detected_results = []


        # ==========================================
        # VEHICLE DETECTION + TRACKING
        # ==========================================

        vehicle_results = (
            vehicle_model.track(
                frame,
                persist=True,
                conf=VEHICLE_CONFIDENCE,
                tracker="bytetrack.yaml",
                verbose=False
            )
        )

        result = vehicle_results[0]


        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):

            return (
                output_frame,
                detected_results
            )


        boxes = (
            result.boxes
            .xyxy
            .cpu()
            .numpy()
        )


        if result.boxes.id is not None:

            track_ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

        else:

            track_ids = [
                -1
            ] * len(boxes)


        frame_height = (
            frame.shape[0]
        )

        frame_width = (
            frame.shape[1]
        )


        # ==========================================
        # PROCESS EACH VEHICLE
        # ==========================================

        for vehicle_box, track_id in zip(
            boxes,
            track_ids
        ):

            vx1, vy1, vx2, vy2 = (
                vehicle_box
                .astype(int)
            )


            # Keep coordinates inside the frame

            vx1 = max(
                0,
                vx1
            )

            vy1 = max(
                0,
                vy1
            )

            vx2 = min(
                frame_width,
                vx2
            )

            vy2 = min(
                frame_height,
                vy2
            )


            # Crop vehicle

            vehicle_crop = frame[
                vy1:vy2,
                vx1:vx2
            ]


            if (
                vehicle_crop.size == 0
            ):

                continue


            # Draw vehicle box

            cv2.rectangle(
                output_frame,
                (vx1, vy1),
                (vx2, vy2),
                (255, 0, 0),
                2
            )


            # ======================================
            # PLATE DETECTION
            # ======================================

            plate_results = (
                plate_model.predict(
                    source=vehicle_crop,
                    conf=PLATE_CONFIDENCE,
                    imgsz=640,
                    verbose=False
                )
            )

            plate_result = (
                plate_results[0]
            )


            if (
                plate_result.boxes
                is not None
                and len(
                    plate_result.boxes
                ) > 0
            ):

                # Select highest-confidence plate

                best_index = (
                    plate_result
                    .boxes
                    .conf
                    .argmax()
                    .item()
                )


                px1, py1, px2, py2 = (
                    plate_result
                    .boxes
                    .xyxy[
                        best_index
                    ]
                    .cpu()
                    .numpy()
                    .astype(int)
                )


                plate_width = (
                    px2 - px1
                )

                plate_height = (
                    py2 - py1
                )


                # ==================================
                # PLATE SIZE FILTER
                # ==================================

                if (
                    plate_width
                    >= MIN_PLATE_WIDTH
                    and plate_height
                    >= MIN_PLATE_HEIGHT
                ):

                    aspect_ratio = (
                        plate_width
                        / plate_height
                    )


                    # ==============================
                    # PLATE SHAPE FILTER
                    # ==============================

                    if (
                        MIN_ASPECT_RATIO
                        <= aspect_ratio
                        <= MAX_ASPECT_RATIO
                    ):

                        # Convert plate coordinates
                        # to original frame coordinates

                        x1 = vx1 + px1

                        y1 = vy1 + py1

                        x2 = vx1 + px2

                        y2 = vy1 + py2


                        # Keep coordinates inside frame

                        x1 = max(
                            0,
                            x1
                        )

                        y1 = max(
                            0,
                            y1
                        )

                        x2 = min(
                            frame_width,
                            x2
                        )

                        y2 = min(
                            frame_height,
                            y2
                        )


                        # Draw plate box

                        cv2.rectangle(
                            output_frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )


                        # ==============================
                        # OCR EVERY 5 FRAMES
                        # ==============================

                        if (
                            self.frame_number
                            % self.ocr_interval
                            == 0
                        ):

                            plate_crop = (
                                frame[
                                    y1:y2,
                                    x1:x2
                                ]
                            )


                            if (
                                plate_crop.size > 0
                            ):

                                plate_text = (
                                    read_plate(
                                        plate_crop
                                    )
                                )


                                if (
                                    plate_text
                                    is not None
                                ):

                                    self.plate_readings[
                                        track_id
                                    ].append(
                                        plate_text
                                    )


            # ======================================
            # GET STABLE PLATE RESULT
            # ======================================

            best_plate = (
                self.get_best_plate(
                    track_id
                )
            )


            if best_plate:

                label = (
                    f"ID {track_id}: "
                    f"{best_plate}"
                )

            else:

                label = (
                    f"ID {track_id}: "
                    "Reading..."
                )


            # Draw label

            cv2.putText(
                output_frame,
                label,
                (
                    vx1,
                    max(
                        25,
                        vy1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            # ======================================
            # SAVE CURRENT RESULT
            # ======================================

            detected_results.append(
                {
                    "vehicle_id": int(
                        track_id
                    ),

                    "license_plate": (
                        best_plate
                        if best_plate
                        else "Reading..."
                    ),

                    "number_of_readings": len(
                        self.plate_readings[
                            track_id
                        ]
                    )
                }
            )


        return (
            output_frame,
            detected_results
        )


    # ==============================================
    # GET FINAL RESULTS
    # ==============================================

    def get_final_results(self):

        final_results = []

        for track_id in sorted(
            self.plate_readings.keys()
        ):

            best_plate = (
                self.get_best_plate(
                    track_id
                )
            )

            if best_plate is None:

                continue

            final_results.append(
                {
                    "vehicle_id": int(
                        track_id
                    ),

                    "license_plate": (
                        best_plate
                    ),

                    "number_of_readings": len(
                        self.plate_readings[
                            track_id
                        ]
                    )
                }
            )

        return final_results

# ==================================================
# PROCESS A SINGLE IMAGE
# ==================================================

def process_image(image):

    output_image = image.copy()

    detected_results = []

    # Detect vehicles in the image
    vehicle_results = vehicle_model.predict(
        source=image,
        conf=VEHICLE_CONFIDENCE,
        verbose=False
    )

    result = vehicle_results[0]

    # No vehicle found
    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):

        return (
            output_image,
            detected_results
        )

    vehicle_boxes = (
        result.boxes
        .xyxy
        .cpu()
        .numpy()
    )

    image_height = image.shape[0]

    image_width = image.shape[1]

    # Process every vehicle
    for vehicle_number, vehicle_box in enumerate(
        vehicle_boxes,
        start=1
    ):

        vx1, vy1, vx2, vy2 = (
            vehicle_box
            .astype(int)
        )

        # Keep vehicle coordinates inside image
        vx1 = max(0, vx1)

        vy1 = max(0, vy1)

        vx2 = min(
            image_width,
            vx2
        )

        vy2 = min(
            image_height,
            vy2
        )

        vehicle_crop = image[
            vy1:vy2,
            vx1:vx2
        ]

        if vehicle_crop.size == 0:

            continue

        # Draw vehicle box
        cv2.rectangle(
            output_image,
            (vx1, vy1),
            (vx2, vy2),
            (255, 0, 0),
            2
        )

        # Detect licence plate inside vehicle
        plate_results = plate_model.predict(
            source=vehicle_crop,
            conf=PLATE_CONFIDENCE,
            imgsz=640,
            verbose=False
        )

        plate_result = plate_results[0]

        plate_text = None

        # Check whether a plate was detected
        if (
            plate_result.boxes is not None
            and len(plate_result.boxes) > 0
        ):

            # Select the highest-confidence plate
            best_index = (
                plate_result
                .boxes
                .conf
                .argmax()
                .item()
            )

            px1, py1, px2, py2 = (
                plate_result
                .boxes
                .xyxy[best_index]
                .cpu()
                .numpy()
                .astype(int)
            )

            plate_width = px2 - px1

            plate_height = py2 - py1

            # Check plate size
            if (
                plate_width >= MIN_PLATE_WIDTH
                and plate_height >= MIN_PLATE_HEIGHT
            ):

                aspect_ratio = (
                    plate_width
                    / plate_height
                )

                # Check plate shape
                if (
                    MIN_ASPECT_RATIO
                    <= aspect_ratio
                    <= MAX_ASPECT_RATIO
                ):

                    # Convert plate coordinates
                    # from vehicle crop to full image

                    x1 = vx1 + px1

                    y1 = vy1 + py1

                    x2 = vx1 + px2

                    y2 = vy1 + py2

                    # Keep coordinates inside image

                    x1 = max(0, x1)

                    y1 = max(0, y1)

                    x2 = min(
                        image_width,
                        x2
                    )

                    y2 = min(
                        image_height,
                        y2
                    )

                    plate_crop = image[
                        y1:y2,
                        x1:x2
                    ]

                    # Run OCR immediately
                    # because an image has only one frame

                    if plate_crop.size > 0:

                        plate_text = read_plate(
                            plate_crop
                        )

                    # Draw plate box

                    cv2.rectangle(
                        output_image,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

        # Create output label

        if plate_text:

            label = (
                f"Vehicle {vehicle_number}: "
                f"{plate_text}"
            )

        else:

            label = (
                f"Vehicle {vehicle_number}: "
                "Plate Not Read"
            )

        # Draw label

        cv2.putText(
            output_image,
            label,
            (
                vx1,
                max(
                    25,
                    vy1 - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Store result

        detected_results.append(
            {
                "vehicle_id": (
                    vehicle_number
                ),

                "license_plate": (
                    plate_text
                    if plate_text
                    else "Not Read"
                ),

                "number_of_readings": (
                    1
                    if plate_text
                    else 0
                )
            }
        )

    return (
        output_image,
        detected_results
    )