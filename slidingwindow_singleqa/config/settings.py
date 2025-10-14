import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from common_utils.paths import bge_model_path
from common_utils.paths import qwen_model_path
from common_utils import config

class Settings:
    WINDOW_SIZE = 5
    STD_DEV_FACTOR = 1.5
    EMBEDDING_MODEL_PATH = bge_model_path
    LLM_MODEL_PATH = qwen_model_path
    max_length = 131072
    max_new_tokens = 8192
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "do_sample": True
    }

    def get_prompt_template(self, segment_text: str, num_questions: int):
        # Language settings
        lang_code = config.LANGUAGE_CODE   # e.g., 'zh'
        lang_name = config.LANGUAGE_NAME
        # Strong instruction to keep output in the target language
        lang_guard = (
            "IMPORTANT: Reply strictly in {name}. "
            "Do not use English unless an English word appears verbatim in the input."
        ).format(name=lang_name)

        # Build prompt
        system_prompt = (f"You are a {lang_name} expert content analyst. Your task is to read a long transcript and identify potential questions that cover the most important educational and mentorship value."
                        f"{lang_guard}"
                    )

        prompt = f"""Read the following transcript carefully and identify the {num_questions} most important questions in {lang_name} providing only educational and mentorship value from this transcript segment. For each question:

        1. Ensure the question captures a key concept or important information from the transcript
        2. Provide a clear, accurate answer to the question based only on information in the transcript
        3. Make sure questions and answers cover different aspects of the content the whole transcript and don't overlap significantly
        4. Select questions and answers in a balanced way from throughout the entire content, not concentrating too heavily on any single section or part
        5. Answers should be in proper detail length and include only the relevant information answering the question properly with educational/mentorship value.
        6. Avoid trivial or overly specific questions.
        7. Use the same Language as of the original content.

        Strictly Format your response as a list of question-answer pairs, with each pair clearly marked (e.g., "Question 1:", "Answer 1:"). Strictly to make your response as structured as possible so it can be easily parsed. Also avoid any other extra words in the start and beginning and only the strict structured response.

        Transcript:
        {segment_text}"""

        return system_prompt, prompt