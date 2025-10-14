import re
from typing import List, Dict

class QuestionParser:
    def parse_selector_response(self, response_text: str) -> List[int]:
        """Parses a string of numbers into a list of integers."""
        numbers = re.findall(r'\d+', response_text)
        return [int(num) for num in numbers]