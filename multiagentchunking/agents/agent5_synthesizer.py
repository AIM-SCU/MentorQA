import torch
from common_utils import config

class Synthesizer:
    def __init__(self, model_handler):
        self.model_handler = model_handler

    def run_agent5_synthesizer(self, question: str, context: str) -> str:
        """Agent 5 (Synthesizer): Answers a single question based on its context."""
        lang_code = config.LANGUAGE_CODE   # e.g., 'zh'
        lang_name = config.LANGUAGE_NAME
        # Strong instruction to keep output in the target language
        lang_guard = (
            "IMPORTANT: Reply strictly in {name}. "
            "Do not use English unless an English word appears verbatim in the input."
        ).format(name=lang_name)
        
        system_prompt = f"You are an expert {lang_name} content analyst. Your task is to read a long transcript and provide answers in {lang_name} that cover the most important educational and mentorship value for the mentioned questions. {lang_guard}"
        
        prompt = f"""**Context:**
        {context}

        ---

        **Question:**
        {question}

        """
        messages = [{"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": prompt}]

        text = self.model_handler.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        inputs = self.model_handler.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=131072).to(self.model_handler.model.device)

        with torch.no_grad():
            outputs = self.model_handler.model.generate(
                **inputs, max_new_tokens=1024, pad_token_id=self.model_handler.tokenizer.eos_token_id)

        answer = self.model_handler.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return answer.strip()