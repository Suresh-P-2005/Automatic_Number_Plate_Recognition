from pathlib import Path

import cv2

from anpr_processor import (
    process_image
)


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

INPUT_IMAGE = (
    BASE_DIR
    / "test_images"
    / "test.jpg"
)

OUTPUT_IMAGE = (
    BASE_DIR
    / "outputs"
    / "images"
    / "result.jpg"
)


image = cv2.imread(
    str(INPUT_IMAGE)
)

if image is None:

    print(
        "Image not found."
    )

    raise SystemExit


processed_image, results = (
    process_image(
        image
    )
)

cv2.imwrite(
    str(OUTPUT_IMAGE),
    processed_image
)

print(
    "Image processing completed."
)

print(
    "Detected results:"
)

for result in results:

    print(result)

print(
    f"Saved to: "
    f"{OUTPUT_IMAGE}"
)