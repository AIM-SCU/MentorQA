#from agents.base_agent import LLMAgent
from processors.qaparser import QAParser
from typing import List, Dict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from common_utils import config

class QAGenerationAgent:
    def __init__(self, base_agent):
        self.base_agent = base_agent

    def run_agent2_qa_generation(self, segment_text: str, num_questions: int) -> List[Dict]:
        """Agent 2: Generate QA pairs for a topic segment"""
    
        # Debug: Show what we're sending to the model
        # print(f"    📝 Segment text length: {len(segment_text)} chars")
        # print(f"    📝 Segment preview: {segment_text[:200]}...")
        # print(f"    🔢 Requested {num_questions} QA pairs")

        # Language settings
        lang_code = config.LANGUAGE_CODE   # e.g., 'zh'
        lang_name = config.LANGUAGE_NAME
        # Strong instruction to keep output in the target language
        lang_guard = (
            "IMPORTANT: Generate exactly reuqested number of questions and answers strictly in {name}. "
            "Do not use English unless an English word appears verbatim in the input."
        ).format(name=lang_name)

        system_prompt = (f"You are a {lang_name} expert content analyst. Your task is to read a long transcript and identify potential questions that cover the most important educational and mentorship value."
                        f"{lang_guard}")
        
        prompt = f"""Identify exactly {num_questions} most important questions in {lang_name} providing only educational and mentorship value from this transcript segment. For each:

1. Ensure the question captures a key concept or important information from the transcript segment
2. Provide a clear, accurate answer to the question based only on information in the transcript segment
3. Make sure questions and answers cover different aspects of the content the whole transcript segment and don't overlap significantly
4. Select questions and answers in a balanced way from throughout the entire content, not concentrating too heavily on any small section or part or section
5. Answers should be in proper detail length and include only the relevant information answering the question properly with educational/mentorship value.
6. Avoid trivial or overly specific questions.
7. Use the same Language as of the original content.

Format strictly as:
Question 1: [question text in {lang_name}]
Answer 1: [answer text in {lang_name}]
...

Transcript Segment:
{segment_text}"""
    
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        generation_params = {
                "max_new_tokens": 8192,  # Sufficient for multiple QAs
                "temperature": 0.7,
                "repetition_penalty": 1.05,
            }
        
        response = self.base_agent.generate_response(messages, generation_params)

        # DEBUG: Save and analyze the raw response
        # print(f"    🔍 RAW MODEL RESPONSE:")
        # print(f"    Response length: {len(response)} chars")
        # print(f"    First 500 chars:")
        # print(f"    {response[:500]}")
        # print("    " + "=" * 50)

        return QAParser.parse_qa_pairs(response)