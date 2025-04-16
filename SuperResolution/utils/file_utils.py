import os
import re

def parse_scale_from_filename(filename):
    """Attempts to extract the scale factor (e.g., xN) from a filename."""
    if not filename:
        return None
    basename = os.path.basename(filename)
    match = re.search(r'[._]?x([2-8])(?:[._]|\.pth$|$)', basename, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, IndexError):
            return None
    return None
