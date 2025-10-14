import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.settings import Settings
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from common_utils.paths import qwen_model_path

class QAModel:
    # Initialize
    def __init__(self, model_path = qwen_model_path):
        self.settings = Settings()
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    # Load model
    def load_model(self):
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
        self.model.generation_config.max_length = self.settings.max_length  # Input context
        self.model.generation_config.max_new_tokens = self.settings.max_new_tokens  # Output length

    def generate_qa_pairs(self, transcript, num_questions = 20):
        """Generate QA pairs from transcript"""

        if isinstance(transcript, list):
        # Handle list of dicts like transcript.json
            if transcript and isinstance(transcript[0], dict):
                transcript = [(d.get("content") or d.get("transcript") or "").strip() for d in transcript]
            # Join into single string context
            transcript = " ".join([t for t in transcript if t])

        elif isinstance(transcript, dict):
            transcript = (transcript.get("content") or transcript.get("transcript") or "").strip()

        elif not isinstance(transcript, str):
            transcript = str(transcript).strip()

        inputs = self.tokenizer(transcript, return_tensors="pt", truncation=True, max_length=self.settings.max_length)
        transcript = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        print(f"ℹ️ Using {inputs.input_ids.shape[1]} tokens from transcript")

        #prompt = self.settings.get_prompt_template(transcript)
        system_prompt, prompt = self.settings.get_prompt_template(transcript, num_questions = num_questions)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize and move to GPU
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # Generate QA pairs
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                **self.settings.generation_config,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Extract response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response

    def cleanup(self):
        # Cleanup model resources
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
