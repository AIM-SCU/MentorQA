import sys
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from common_utils.paths import qwen_model_path

class QwenModelHandler:
    def __init__(self, llm_model_path = qwen_model_path):
        self.llm_model_path = llm_model_path
        self.tokenizer = None
        self.model = None
        self.loaded = False

    def load_model(self):
        """Load Qwen model once and reuse"""
        if self.loaded:
            print("✅ Reusing already loaded Qwen model")
            return
        """    
        if self.model is not None and self.tokenizer is not None:
            print("✅ Reusing already loaded Qwen model")
            return self.model, self.tokenizer
        """

        """Load the LLM model and tokenizer"""
        print("⏳ Loading Qwen2.5-7B-Instruct-1M for generation tasks...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).eval()
        self.loaded = True

        # Configure context lengths
        self.model.generation_config.max_length = 131072  # Input context
        self.model.generation_config.max_new_tokens = 8192  # Output length
        
        print("✅ Qwen model loaded successfully")
        #return self.model, self.tokenizer

    def get_model(self):
        """Get the loaded model and tokenizer"""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model, self.tokenizer

    def unload_model(self):
        """Unload the model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.loaded = False

        import gc
        #objs = [obj for obj in gc.get_objects() if torch.is_tensor(obj)]
        #print(f"Still {len(objs)} tensors in memory")
        gc.collect()  # force garbage collection
        torch.cuda.empty_cache()