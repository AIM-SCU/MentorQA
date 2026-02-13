import json
from typing import List, Dict

class JSONParser:
    def extract_json_from_response(self, response: str) -> List[Dict]:
        """Robustly extracts a JSON list from an LLM's text response."""
        start_idx = response.find('[')
        end_idx = response.rfind(']')
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON Decode Error: {e}")
                return None
        return None