from processors.qaparser import QAParser
from typing import List, Dict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from common_utils import config


class TopicSegmentationAgent:
    def __init__(self, base_agent):
        self.base_agent = base_agent

    def run_agent1_topic_segmentation(
        self, numbered_transcript: str, total_lines: int
    ) -> List[Dict]:
        """Identify topics and their line ranges with language support"""
        # Language settings
        lang_code = config.LANGUAGE_CODE  # e.g., 'zh'
        lang_name = config.LANGUAGE_NAME
        # Strong instruction to keep output in the target language
        lang_guard = (
            "IMPORTANT: Reply strictly in {name}. "
            "Do not use English unless an English word appears verbatim in the input."
        ).format(name=lang_name)

        system_prompt = (
            f"You are a {lang_name} expert at analyzing transcripts and segmenting them by topic. "
            "The transcript has been split into numbered lines where each line represents "
            "a complete thought. Identify topic boundaries and assign concise topic titles."
            f"{lang_guard}"
        )

        prompt = f"""Output a JSON list of dictionaries in {lang_name} with these keys:
        - "topic": Concise descriptive title (3-7 words)
        - "start_line": First line number of this topic
        - "end_line": Last line number of this topic

        Rules:
        1. Topics must cover consecutive line numbers
        2. Entire transcript must be covered without gaps or overlaps
        3. The first topic must start at line 1
        4. The last topic must end at line {total_lines}
        5. Use line numbers exactly as provided
        6. Output ONLY the JSON with no additional text
        7. Write topic titles in {lang_name}

        Numbered Transcript:
        {numbered_transcript}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        generation_params = {
            "max_new_tokens": 2048,  # Increased for reliability
            "temperature": 0.3,  # More deterministic for structure
        }

        response = self.base_agent.generate_response(messages, generation_params)
        print("debug", response, flush=True)
        return QAParser.extract_json_from_response(response)
