import torch
from typing import Dict, List
from agents.base_agent import QwenModelHandler
from common_utils import config

class AnswerSynthesizer:
    
    def __init__(self, model_handler):
        self.model_handler = model_handler
        #self.model, self.tokenizer = None, None

    # def prepare(self):
    #     """Load the model once before answering questions"""
    #     #self.model, self.tokenizer = self.model_handler.load_model()
    #     pass

    
    # def load_model(self):
    #     #Load the LLM model and tokenizer
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path)
    #     self.model = AutoModelForCausalLM.from_pretrained(
    #         self.llm_model_path,
    #         device_map="auto",
    #         torch_dtype=torch.bfloat16
    #     ).eval()
    

    def run_agent2_answer_synthesis(self, question: str, context_chunks: List[str]) -> Dict:
        """Agent 2: Use retrieved chunks to answer a single question."""
        model, tokenizer = self.model_handler.get_model()
        
        # Language settings
        lang_code = config.LANGUAGE_CODE   # e.g., 'zh'
        lang_name = config.LANGUAGE_NAME
        # Strong instruction to keep output in the target language
        lang_guard = (
            "IMPORTANT: Reply strictly in {name}. "
            "Do not use English unless an English word appears verbatim in the input."
        ).format(name=lang_name)

        system_prompt = f"You are an expert {lang_name} content analyst. Your task is to read a long transcript and provide answers in {lang_name} that cover the most important educational and mentorship value for the mentioned questions. {lang_guard}"
        
        # Join the chunks to form the context
        context = "\n\n---\n\n".join(context_chunks)
        
        prompt = f"""Please answer the following question in {lang_name} using ONLY the information from the context provided below. {lang_guard}

        Question:
        {question}

        Context:
        {context}

        Answer:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=131072).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id)
            
        answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Basic validation to ensure the model didn't refuse to answer
        if "not available in the provided context" in answer.lower():
            return None

        return {"question": question, "answer": answer}