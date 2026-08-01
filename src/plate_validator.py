import re


def is_valid_plate(text):

    if not text:

        return False

    text = text.upper()

    # Common Indian licence plate format
    #
    # Examples:
    # TN38AB1234
    # KA01CD5678
    # MH12DE1234

    pattern = (
        r"^[A-Z]{2}"
        r"\d{1,2}"
        r"[A-Z]{1,3}"
        r"\d{4}$"
    )

    return (
        re.fullmatch(
            pattern,
            text
        )
        is not None
    )