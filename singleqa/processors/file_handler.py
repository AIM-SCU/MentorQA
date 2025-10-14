import json
import os

class FileHandler:
    def read_transcript(self, file_name):
        # """Read and validate transcript file"""
        # if not os.path.exists(file_name) or not file_name.endswith('.txt'):
        #     raise ValueError(f"File {file_name} not found or not a .txt file")
        #     #print(f"❌ Error: '{file_name}' not found or not a .txt file. Try again.")

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

    def save_qa_pairs(self, qa_data, file_name, suffix):
        """Save QA pairs to JSON file"""
        output_file = f"{suffix}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)
        return output_file
