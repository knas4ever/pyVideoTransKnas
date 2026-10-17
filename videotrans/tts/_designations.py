"""Spell short letter-and-number names so a voice model does not swallow them."""

import re

# R2D2, R2-D2, C-3PO, BB8. A plain word or a year has only letters or only digits.
_DESIGNATION = re.compile(
    r"(?<![0-9A-Za-zÆØÅæøå])"
    r"([0-9A-Za-zÆØÅæøå]+(?:-[0-9A-Za-zÆØÅæøå]+)*)"
    r"(?![0-9A-Za-zÆØÅæøå])"
)


def speakable_text(text):
    """Return text where a short code is spelled, such as R2D2 to R-2-D-2."""

    def spell(match):
        raw = match.group(1)
        parts = raw.split("-")
        chars = [char for char in raw if char.isalnum()]
        if not 2 <= len(chars) <= 8:
            return raw
        if not any(char.isalpha() for char in chars) or not any(char.isdigit() for char in chars):
            return raw
        # R2-enhed is a word with a code in front. A droid name has no letter run longer than two.
        for part in parts:
            run = 0
            for char in part:
                if char.isalpha():
                    run += 1
                    if run > 2:
                        return raw
                else:
                    run = 0
        return "-".join(chars)

    return _DESIGNATION.sub(spell, text or "")
