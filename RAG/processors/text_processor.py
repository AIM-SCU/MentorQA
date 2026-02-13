from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

class TextProcessor:
    def split_text(self, transcript: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
        """Split transcript into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        return splitter.split_text(transcript)