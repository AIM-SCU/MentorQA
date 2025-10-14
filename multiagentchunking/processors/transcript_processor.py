import re
from typing import List

class TranscriptProcessor:
    def number_transcript_lines(self, transcript: str) -> List[str]:
        """Splits transcript into sentences and returns a list of numbered lines."""
        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        numbered_lines = [f"{i}: {sentence.strip()}" for i, sentence in enumerate(sentences, 1) if sentence.strip()]
        return numbered_lines