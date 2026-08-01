from pathlib import Path

from video_processor import (
    process_video
)


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


INPUT_VIDEO = (
    BASE_DIR
    / "videos"
    / "input1.mp4"
)


OUTPUT_VIDEO = (
    BASE_DIR
    / "outputs"
    / "videos"
    / "processed_video.mp4"
)


OUTPUT_CSV = (
    BASE_DIR
    / "outputs"
    / "csv"
    / "video_results.csv"
)


print(
    "Starting ANPR video processing..."
)


result = process_video(
    input_video_path=INPUT_VIDEO,
    output_video_path=OUTPUT_VIDEO,
    csv_output_path=OUTPUT_CSV
)


print(
    "\nFinal result"
)

print(
    "=" * 50
)

print(
    f"Frames processed: "
    f"{result['frames_processed']}"
)

print(
    f"Recognised vehicles: "
    f"{result['total_recognised_vehicles']}"
)

print(
    f"Output video:\n"
    f"{result['output_video']}"
)

print(
    f"CSV file:\n"
    f"{result['csv_file']}"
)


print(
    "\nDetected plates:"
)

for item in result["results"]:

    print(
        item
    )