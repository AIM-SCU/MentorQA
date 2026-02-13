import re
from typing import List

class QuestionParser:
    def parse_generated_questions(self, text: str) -> List[str]:
        """Extracts questions from a numbered list string."""
        questions = []
        for line in text.split('\n'):
            # Use regex to find lines that start with a number and a period, then capture the question.
            match = re.match(r'^\s*\d+\.\s*(.+)', line)
            if match:
                questions.append(match.group(1).strip())
        return questions