from pathlib import Path
import csv
import subprocess
import cv2

from anpr_processor import (
    ANPRSession
)

def convert_to_browser_video(
    input_video_path,
    output_video_path
):

    command = [

        "ffmpeg",

        "-y",

        "-i",
        str(input_video_path),

        "-c:v",
        "libx264",

        "-preset",
        "fast",

        "-crf",
        "23",

        "-pix_fmt",
        "yuv420p",

        "-movflags",
        "+faststart",

        "-an",

        str(output_video_path)
    ]


    subprocess.run(
        command,
        check=True
    )

def process_video(
    input_video_path,
    output_video_path,
    csv_output_path
):

    # Convert paths to Path objects

    input_video_path = Path(
        input_video_path
    )

    output_video_path = Path(
        output_video_path
    )

    csv_output_path = Path(
        csv_output_path
    )


    # Create output folders

    output_video_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # Open input video

    cap = cv2.VideoCapture(
        str(input_video_path)
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{input_video_path}"
        )


    # Read video information

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

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    # Use a fallback FPS if needed

    if fps <= 0:

        fps = 24


    print(
        f"Video size: "
        f"{width} x {height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Total frames: "
        f"{total_frames}"
    )


    # Create output video

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )


    temporary_video_path = (
        output_video_path
        .with_name(
            output_video_path.stem
            + "_temporary.mp4"
        )
    )


    out = cv2.VideoWriter(
        str(
            temporary_video_path
        ),
        fourcc,
        fps,
        (
            width,
            height
        )
    )


    if not out.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create output video."
        )


    # Create one ANPR session
    # This preserves tracking and OCR readings

    anpr = ANPRSession()


    frame_count = 0


    print(
        "\nProcessing video..."
    )


    # Process every frame

    while True:

        success, frame = cap.read()


        if not success:

            break


        frame_count += 1


        processed_frame, _ = (
            anpr.process_frame(
                frame
            )
        )


        out.write(
            processed_frame
        )


        # Show progress every 30 frames

        if (
            frame_count % 30
            == 0
        ):

            print(
                f"Processed "
                f"{frame_count}/"
                f"{total_frames} frames"
            )


    # Release resources

    cap.release()

    out.release()


    # Convert OpenCV output to
    # browser-compatible H.264 MP4

    print(
        "\nConverting video "
        "for browser playback..."
    )


    convert_to_browser_video(
        input_video_path=(
            temporary_video_path
        ),
        output_video_path=(
            output_video_path
        )
    )


    # Remove temporary video

    if temporary_video_path.exists():

        temporary_video_path.unlink()


    # Get final OCR-voted results

    final_results = (
        anpr.get_final_results()
    )


    # Save results to CSV

    with open(
        csv_output_path,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "Vehicle_ID",
                "License_Plate",
                "Number_of_Readings"
            ]
        )


        writer.writeheader()


        for result in final_results:

            writer.writerow(
                {
                    "Vehicle_ID": (
                        result[
                            "vehicle_id"
                        ]
                    ),

                    "License_Plate": (
                        result[
                            "license_plate"
                        ]
                    ),

                    "Number_of_Readings": (
                        result[
                            "number_of_readings"
                        ]
                    )
                }
            )


    print(
        "\nVideo processing completed."
    )


    # Return results to the caller

    return {

        "success": True,

        "frames_processed": (
            frame_count
        ),

        "total_recognised_vehicles": len(
            final_results
        ),

        "results": (
            final_results
        ),

        "output_video": str(
            output_video_path
        ),

        "csv_file": str(
            csv_output_path
        )
    }