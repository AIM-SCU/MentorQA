import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from common_utils.paths import qwen_model_path
from common_utils.paths import bge_model_path

class Settings:
    WINDOW_SIZE = 5
    STD_DEV_FACTOR = 1.5
    EMBEDDING_MODEL_PATH = bge_model_path
    LLM_MODEL_PATH = qwen_model_path
    MAX_QUESTIONS = 20