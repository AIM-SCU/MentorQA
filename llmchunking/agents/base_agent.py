import sys
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from common_utils.paths import qwen_model_path

class LLMAgent:
    def __init__(self, model_path = qwen_model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        #self.loaded = False

    def load_model(self):
        """Load the LLM model and tokenizer"""
        print("⏳ Loading Qwen2.5-7B-Instruct-1M model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval()

        # Configure context lengths
        self.model.generation_config.max_length = 131072  # Input context
        self.model.generation_config.max_new_tokens = 8192  # Output length

        print("✅ Model loaded successfully")
        #self.loaded = True

    def generate_response(self, messages: List[Dict], generation_params: Optional[Dict] = None) -> str:
        """Generate a response from the LLM"""
        #if not self.loaded:
        #    raise RuntimeError("Model not loaded. Call load_model() first.")

        default_params = {
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        if generation_params:
            default_params.update(generation_params)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=131072).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **default_params
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response

    def unload_model(self):
        """Unload the model to free memory"""
        #if self.loaded:
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        #    self.loaded = False
        #    print("✅ Model unloaded and memory freed")
