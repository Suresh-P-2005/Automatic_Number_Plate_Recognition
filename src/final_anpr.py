from pathlib import Path
from collections import defaultdict, Counter
import csv
import cv2
import easyocr

from ultralytics import YOLO

from ocr_utils import (
    correct_plate,
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

VIDEO_PATH = (
    BASE_DIR
    / "videos"
    / "input1.mp4"
)

OUTPUT_PATH = (
    BASE_DIR
    / "outputs"
    / "final_anpr_output.mp4"
)

CSV_OUTPUT_PATH = (
    BASE_DIR
    / "outputs"
    / "anpr_results.csv"
)

# ==================================================
# SETTINGS
# ==================================================

VEHICLE_CONFIDENCE = 0.40

PLATE_CONFIDENCE = 0.45

OCR_INTERVAL = 3

MIN_PLATE_WIDTH = 40

MIN_PLATE_HEIGHT = 10

MIN_ASPECT_RATIO = 1.5

MAX_ASPECT_RATIO = 8.0


# ==================================================
# LOAD MODELS
# ==================================================

print(
    "Loading vehicle model..."
)

vehicle_model = YOLO(
    str(VEHICLE_MODEL_PATH)
)

print(
    "Loading plate model..."
)

plate_model = YOLO(
    str(PLATE_MODEL_PATH)
)

print(
    "Loading EasyOCR..."
)

ocr_reader = easyocr.Reader(
    ["en"],
    gpu=False
)

print(
    "All models loaded."
)


# ==================================================
# OCR MEMORY
# ==================================================

# Stores all valid OCR readings
# for each vehicle tracking ID

plate_readings = defaultdict(
    list
)


# ==================================================
# OPEN VIDEO
# ==================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():

    print(
        "Error: Cannot open video."
    )

    raise SystemExit


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:

    fps = 24


# ==================================================
# OUTPUT VIDEO
# ==================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)


# ==================================================
# HELPER FUNCTIONS
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

    results = (
        ocr_reader.readtext(
            enhanced,
            detail=1,
            paragraph=False,
            allowlist=(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
            )
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


def get_best_plate(track_id):

    readings = (
        plate_readings[
            track_id
        ]
    )

    if not readings:

        return None

    counts = Counter(
        readings
    )

    best_text, best_count = (
        counts.most_common(1)[0]
    )

    if best_count >= 2:
        return best_text


    return best_text

def print_final_summary():

    total_tracked_vehicles = len(
        plate_readings
    )

    recognised_vehicles = 0

    unrecognised_vehicles = 0

    print("\n" + "=" * 60)

    print("FINAL ANPR RESULTS")

    print("=" * 60)

    for track_id in sorted(
        plate_readings.keys()
    ):

        best_plate = get_best_plate(
            track_id
        )

        reading_count = len(
            plate_readings[track_id]
        )

        if best_plate is not None:

            recognised_vehicles += 1

            print(
                f"Vehicle ID: {track_id}"
            )

            print(
                f"Plate: {best_plate}"
            )

            print(
                f"Valid OCR readings: "
                f"{reading_count}"
            )

        else:

            unrecognised_vehicles += 1

            print(
                f"Vehicle ID: {track_id}"
            )

            print(
                "Plate: Not recognised"
            )

            print(
                f"Valid OCR readings: "
                f"{reading_count}"
            )

        print("-" * 60)

    print(
        f"Total tracked vehicles: "
        f"{total_tracked_vehicles}"
    )

    print(
        f"Vehicles with recognised plates: "
        f"{recognised_vehicles}"
    )

    print(
        f"Vehicles without recognised plates: "
        f"{unrecognised_vehicles}"
    )

    print("=" * 60)
# ==================================================
# PROCESS VIDEO
# ==================================================

frame_number = 0


while cap.isOpened():

    success, frame = cap.read()

    if not success:

        break

    frame_number += 1

    output_frame = frame.copy()


    # ----------------------------------------------
    # VEHICLE DETECTION + TRACKING
    # ----------------------------------------------

    vehicle_results = (
        vehicle_model.track(
            frame,
            persist=True,
            conf=VEHICLE_CONFIDENCE,
            tracker="bytetrack.yaml",
            verbose=False
        )
    )

    result = (
        vehicle_results[0]
    )


    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):

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


        # ------------------------------------------
        # PROCESS EACH VEHICLE
        # ------------------------------------------

        for vehicle_box, track_id in zip(
            boxes,
            track_ids
        ):

            vx1, vy1, vx2, vy2 = (
                vehicle_box
                .astype(int)
            )

            vx1 = max(
                0,
                vx1
            )

            vy1 = max(
                0,
                vy1
            )

            vx2 = min(
                width,
                vx2
            )

            vy2 = min(
                height,
                vy2
            )

            vehicle_crop = frame[
                vy1:vy2,
                vx1:vx2
            ]

            if (
                vehicle_crop.size == 0
            ):

                continue


            # --------------------------------------
            # PLATE DETECTION
            # --------------------------------------

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


            # Draw vehicle box
            cv2.rectangle(
                output_frame,
                (vx1, vy1),
                (vx2, vy2),
                (255, 0, 0),
                2
            )


            # --------------------------------------
            # PLATE FOUND
            # --------------------------------------

            if (
                plate_result.boxes
                is not None
                and len(
                    plate_result.boxes
                ) > 0
            ):

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


                # ----------------------------------
                # FILTER INVALID PLATES
                # ----------------------------------

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

                    if (
                        MIN_ASPECT_RATIO
                        <= aspect_ratio
                        <= MAX_ASPECT_RATIO
                    ):

                        # Convert to full-frame
                        # coordinates

                        x1 = vx1 + px1
                        y1 = vy1 + py1

                        x2 = vx1 + px2
                        y2 = vy1 + py2


                        # Draw plate box

                        cv2.rectangle(
                            output_frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )


                        # ----------------------------------
                        # OCR EVERY 5 FRAMES
                        # ----------------------------------

                        if (
                            frame_number
                            % OCR_INTERVAL
                            == 0
                        ):

                            plate_crop = (
                                frame[
                                    y1:y2,
                                    x1:x2
                                ]
                            )

                            if (
                                plate_crop
                                .size > 0
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

                                    plate_readings[
                                        track_id
                                    ].append(
                                        plate_text
                                    )


            # --------------------------------------
            # GET MOST COMMON OCR RESULT
            # --------------------------------------

            best_plate = (
                get_best_plate(
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


            # Draw result

            cv2.putText(
                output_frame,
                label,
                (
                    vx1,
                    max(
                        25,
                        vy1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


    # ----------------------------------------------
    # SAVE AND SHOW
    # ----------------------------------------------

    out.write(
        output_frame
    )

    cv2.imshow(
        "Final ANPR",
        output_frame
    )

    if (
        cv2.waitKey(1)
        & 0xFF
        == ord("q")
    ):

        break


# ==================================================
# CLEAN UP
# ==================================================

cap.release()

out.release()

cv2.destroyAllWindows()


# ==================================================
# SAVE FINAL RESULTS TO CSV
# ==================================================

with open(
    CSV_OUTPUT_PATH,
    mode="w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.writer(
        csv_file
    )

    # CSV column names
    writer.writerow([
        "Vehicle_ID",
        "License_Plate",
        "Number_of_Readings"
    ])

    # Save one final result for each vehicle
    for track_id in sorted(
        plate_readings.keys()
    ):

        best_plate = get_best_plate(
            track_id
        )

        # Skip vehicles without a stable result
        if best_plate is None:
            continue

        number_of_readings = len(
            plate_readings[track_id]
        )

        writer.writerow([
            track_id,
            best_plate,
            number_of_readings
        ])


# ==================================================
# FINAL MESSAGE
# ==================================================

print(
    "\nFinal ANPR completed."
)

print(
    f"\nVideo saved to:\n"
    f"{OUTPUT_PATH}"
)

print(
    f"\nCSV saved to:\n"
    f"{CSV_OUTPUT_PATH}"
)
print_final_summary()