import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
memory_root = Path(__file__).parent.parent

# GPU Memory Function path
gpu_memory_path = memory_root / "common_utils"

# Model paths
qwen_model_path = project_root / "Qwen2.5-7B-Instruct-1M"
bge_model_path = project_root / "BGE-M3"

#project_root = os.path.dirname(os.path.abspath(__file__))
#model_path = os.path.join(project_root, "Qwen model is here")
