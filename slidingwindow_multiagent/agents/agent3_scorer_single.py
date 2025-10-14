import re
import torch

class Scorer:
    def __init__(self, model_handler):
        self.model_handler = model_handler

    def run_agent3_scorer_single(self, question: str, context_hint: str) -> float:
        """
        Score a single question 1-10 for educational/mentorship value.
        Keep it short so we can call it per-question (reduces cross-item bias).
        """
        print("🧠 Running Agent 3 (Scorer) to rating every questions...")
        
        system_prompt = "You are an expert content evaluator. Return ONLY a number 1-10."
        prompt = f"""Rate the following question for educational/mentorship value (1=poor, 10=excellent). 
        Question: {question}
        Hint: {context_hint}. 
        Return just a number."""
        
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]

        text = self.model_handler.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        inputs = self.model_handler.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=131072).to(self.model_handler.model.device)

        with torch.no_grad():
            outputs = self.model_handler.model.generate(
                **inputs, max_new_tokens=10, pad_token_id=self.model_handler.tokenizer.eos_token_id)

        response = self.model_handler.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        m = re.search(r'(\d+)', response)
        return min(10.0, max(1.0, float(m.group(1)))) if m else 5.0