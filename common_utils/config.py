# config.py
# Global, process-wide config values shared across modules.

# default language
LANGUAGE_CODE: str = "en"
LANGUAGE_NAME: str = "English"

# Allowed languages
LANGUAGE_MAP = {
    "en": ("en", "English"),
    "eng": ("en", "English"),
    "english": ("en", "English"),

    "hi": ("hi", "Hindi"),
    "hind": ("hi", "Hindi"),
    "hindi": ("hi", "Hindi"),

    "zh": ("zh", "Chinese"),
    "cn": ("zh", "Chinese"),
    "ch": ("zh", "Chinese"),
    "chinese": ("zh", "Chinese"),

    "ro": ("ro", "Romanian"),
    "romanian": ("ro", "Romanian"),
}
