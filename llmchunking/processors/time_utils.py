def format_hhmmss(seconds: float) -> str:
    """Format seconds to HH:MM:SS (no ms)"""
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"