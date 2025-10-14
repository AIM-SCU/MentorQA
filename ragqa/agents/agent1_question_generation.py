import torch
from typing import List
from processors.question_parser import QuestionParser
from agents.base_agent import QwenModelHandler
from common_utils import config

class QuestionGenerator:
    def __init__(self, model_handler, question_parser):
        self.model_handler = model_handler
        self.question_parser = question_parser

    
    # def load_model(self):
    #     #Load the LLM model and tokenizer
    #     print("⏳ Loading Qwen2.5-7B-Instruct-1M for generation tasks...")
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path)
    #     self.model = AutoModelForCausalLM.from_pretrained(
    #         self.llm_model_path,
    #         device_map="auto",
    #         torch_dtype=torch.bfloat16
    #     ).eval()
    

    def run_agent1_question_generation(self, transcript: str) -> List[str]:
        """Agent 1: Read the full transcript and generate potential questions."""
        
        model, tokenizer = self.model_handler.get_model()
        
        # Language settings
        lang_code = config.LANGUAGE_CODE   # e.g., 'zh'
        lang_name = config.LANGUAGE_NAME
        # Strong instruction to keep output in the target language
        lang_guard = (
            "IMPORTANT: Reply strictly in {name}. "
            "Do not use English unless an English word appears verbatim in the input."
        ).format(name=lang_name)
        
        system_prompt = f"You are an expert {lang_name} content analyst. Your task is to read a long transcript and identify potential questions in {lang_name} that cover the most important educational and mentorship value. {lang_guard}"
        
        prompt = f"""Based on the following transcript, generate a list of exactly 20 diverse and high-value questions in {lang_name}. {lang_guard}

        Guidelines for questions:
        - Focus on key concepts, advice, and actionable insights.
        - Ensure questions span the entire transcript, from beginning to end.
        - Avoid trivial or overly specific questions.
        - Phrase them as clear, standalone questions.

        Format your output STRICTLY as a numbered list (e.g., "1. What is the core philosophy behind...?"). Do not add any other text before or after the list.

        Transcript:
        {transcript}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=131072).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id)
        
        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        print("testing", response_text)

        return self.question_parser.parse_generated_questions(response_text)