import os
import json

class FileHandler:
    def read_transcript(self, file_name):
        """Read and validate the transcript file name"""
        # if not os.path.exists(file_name):
        #     print(f"❌ Error: '{file_name}' not found. Please check the path and try again.")

        # # Read and truncate transcript
        # print(f"⏳ Reading content from '{file_name}'...")
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

    def save_json(self, data, file_name, suffix):
        base_name = os.path.splitext(file_name)[0]
        output_file = f"{suffix}.json"
        print(f"💾 Saving results to '{output_file}'...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return output_file
