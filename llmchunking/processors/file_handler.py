import os
import json
from typing import List,Dict

class FileHandler:
    def get_transcript_file(self, file_name):
        """Get transcript filename from flag or prompt"""
        #file_name = str(file_name).strip()
        if not os.path.exists(file_name) or not file_name.lower().endswith('.json'):
            raise ValueError(f"File {file_name} not found or not a .json file")

        # Read transcript JSON (segments with start/end/text)
        print(f"⏳ Reading {file_name}...")
        with open(file_name, 'r', encoding='utf-8') as f:
            segments_json = json.load(f)
            if not isinstance(segments_json, list):
                raise ValueError("Transcript JSON must be a list of segment objects.")
            return segments_json

        # Below code gives error because the file_name was not passed. Using the 
        # code from file_handler in SingleQA folder.
        
        # if transcript_file:   # ✅ first check flag-based file
        #     print(f"\n📂 Using transcript file from flag: {transcript_file}")
        #     if os.path.exists(transcript_file) and transcript_file.endswith(".txt"):
        #         return transcript_file
        #     else:
        #         raise FileNotFoundError(f"Transcript file not found: {transcript_file}")

        # # ✅ fallback: interactive prompt (unchanged)
        # if not os.path.exists(file_name) or not file_name.endswith(".txt"):
        #     print(f"❌ Error: '{file_name}' not found or not a .txt file. Try again.")


    def save_output(self, data: Dict, file_name: str, suffix: str) -> str:
        """Save data to JSON file"""
        base_name = os.path.splitext(file_name)[0]
        output_file = f"{suffix}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return output_file
