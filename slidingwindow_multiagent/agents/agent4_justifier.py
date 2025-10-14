import torch
from typing import Dict
from common_utils import config

class Justifier:
    def __init__(self, model_handler):
        self.model_handler = model_handler

    def run_agent4_justifier(self, question_item: Dict) -> str:
        """Agent 4 (Justifier): Provides a reason for a single question's status."""
        lang_code = config.LANGUAGE_CODE   # e.g., 'zh'
        lang_name = config.LANGUAGE_NAME
        
        system_prompt = f"You are a {lang_name} content analyst. Your task is to provide a clear and concise justification for a question's selection status in {lang_name}."
        
        prompt = f"""You will be given a question, its source topic, and its "Selected" or "Rejected" status. Your job is to explain *why* that status makes sense.

        **Question:** {question_item['question']}
        **Selection Status:** {question_item['status']}

        Based on the information above, provide a concise reason in {lang_name}. Use the following as **sample reasons, but you are not limited to them**:
        - **Good `Selected` reasons:**
            - "Providing high educational value and mentorship value."
            - "Asks about career transition challenges, which has direct mentorship value
            - "Addresses the key concept discussed in the source."
        - **Good `Rejected` reasons:**
            - "The question is too basic and offers little insight."
            - "This topic is already covered by another, more specific selected question."
            - "This question has a low score based on the result of agent 3."

        **Reason:**"""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        text = self.model_handler.tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        inputs = self.model_handler.tokenizer(text, return_tensors="pt", truncation=True, max_length=131072).to(self.model_handler.model.device)
        with torch.no_grad():
            outputs = self.model_handler.model.generate(**inputs, max_new_tokens=128,
                                    pad_token_id=self.model_handler.tokenizer.eos_token_id)
        reason = self.model_handler.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return reason.strip()