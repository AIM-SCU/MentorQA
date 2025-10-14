import re
import torch
from typing import List
from common_utils import config

class Inquisitor:
    def __init__(self, model_handler):
        self.model_handler = model_handler

    def run_agent2_inquisitor(self, segment_content: str) -> List[str]:
        """Agent 2 (Inquisitor): Brainstorms questions for a single text segment."""
        lang_code = config.LANGUAGE_CODE   # e.g., 'zh'
        lang_name = config.LANGUAGE_NAME
        # Strong instruction to keep output in the target language
        lang_guard = (
            "IMPORTANT: Reply strictly in {name}. "
            "Do not use English unless an English word appears verbatim in the input."
        ).format(name=lang_name)
        
        system_prompt = f"You are an expert {lang_name} content analyst. Your task is to read a long transcript and identify potential questions in {lang_name} that cover the most important educational and mentorship value. {lang_guard}"
        
        prompt = f"""Based on the following text, generate a list of potential questions with high educational or mentorship value in {lang_name}. Format your output as a simple numbered list.
        Avoid generating duplicate questions with similar meanings.
        
        Text Segment:
        {segment_content}"""
        
        messages = [{"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": prompt}]

        text = self.model_handler.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        inputs = self.model_handler.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=131072).to(self.model_handler.model.device)

        with torch.no_grad():
            outputs = self.model_handler.model.generate(
                **inputs, max_new_tokens=2048, pad_token_id=self.model_handler.tokenizer.eos_token_id)

        response = self.model_handler.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        questions = [line.strip() for line in re.split(r'\d+\.\s*', response) if line.strip()]
        return questions