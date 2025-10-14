import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from common_utils.paths import qwen_model_path

class QAModel:
    def __init__(self, model_path = qwen_model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the LLM model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
            ).eval()
            
        # Configure context lengths
        self.model.generation_config.max_length = 131072  # Input context
        self.model.generation_config.max_new_tokens = 8192  # Output length
        return self

    def cleanup(self):
        # Cleanup model resources
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()