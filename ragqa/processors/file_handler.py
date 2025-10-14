import os
import json
from typing import List, Dict

class FileHandler:
    def get_transcript_file(self, file_name) -> str:
        # """Read and validate transcript file"""
        # if not os.path.exists(file_name):
        #     print(f"❌ Error: '{file_name}' not found. Try again.")

        # # Read and truncate transcript
        # with open(file_name, 'r', encoding='utf-8') as f:
        #     return f.read()

        if not os.path.exists(file_name) or not file_name.lower().endswith('.json'):
            raise ValueError(f"File {file_name} not found or not a .json file")

        # Read transcript JSON (segments with start/end/text)
        print(f"⏳ Reading {file_name}...")
        with open(file_name, 'r', encoding='utf-8') as f:
            segments_json = json.load(f)
            if not isinstance(segments_json, list):
                raise ValueError("Transcript JSON must be a list of segment objects.")
            return segments_json

    def save_results(self, final_qa_pairs: List[Dict], file_name: str, suffix: str) -> str:
        """Save QA pairs to JSON file"""
        base_name = os.path.splitext(file_name)[0]
        output_file = f"{suffix}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_qa_pairs, f, indent=2, ensure_ascii=False)
        return output_file
