import re


def correct_plate(text):

    # Convert to uppercase
    text = text.upper()

    # Remove spaces and special characters
    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


def is_valid_plate(text):

    # Basic Indian licence plate validation
    # Examples:
    # TN38AB1234
    # KA01CD5678

    pattern = (
        r"^[A-Z]{2}"
        r"\d{1,2}"
        r"[A-Z]{1,3}"
        r"\d{4}$"
    )

    return (
        re.match(
            pattern,
            text
        )
        is not None
    )