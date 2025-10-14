import os
import json
import numpy as np
from scipy.signal import argrelextrema
from typing import Union
from pathlib import Path
import stanza


class NLPUtils:
    def __init__(self, use_gpu: bool = True):
        """
        Handles sentence tokenization (via Stanza),
        cosine similarity, and boundary detection.
        """
        # langs folder is one level above main.py’s folder
        self.root_dir = Path(__file__).resolve().parents[3] / "langs"
        self.use_gpu = use_gpu
        self.pipelines = {}  # cache stanza pipelines per language

    def get_pipeline(self, lang: str):
        """Lazy-load and cache a Stanza pipeline for a given language."""
        if lang not in self.pipelines:
            model_dir = str(self.root_dir / lang)
            self.pipelines[lang] = stanza.Pipeline(
                lang=lang,
                dir=model_dir,          # load from langs/<code>
                use_gpu=self.use_gpu,
                processors="tokenize",
                download_method=None    # prevent auto-downloads
            )
        return self.pipelines[lang]

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculates cosine similarity between two numpy vectors."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)

    def sentence_tokenize(self, transcript_data: Union[str, list], lang: str = "en") -> list:
        """
        Split transcript text into sentences using Stanza.
        transcript_data can be:
          - a list of Whisper segments (dicts with "text")
          - a JSON string encoding such a list
          - a plain text string
        lang: language code ("en", "zh", "hi", "ro", etc.)
        """
        # --- normalize input ---
        if isinstance(transcript_data, list):
            text = " ".join([segment.get("text", "") for segment in transcript_data])
        elif isinstance(transcript_data, str):
            try:
                json_data = json.loads(transcript_data)
                if isinstance(json_data, list):
                    text = " ".join([segment.get("text", "") for segment in json_data])
                else:
                    text = transcript_data
            except json.JSONDecodeError:
                text = transcript_data
        else:
            raise ValueError("Unsupported transcript data format")

        # --- run stanza pipeline ---
        nlp = self.get_pipeline(lang)
        doc = nlp(text)
        return [s.text for s in doc.sentences]

    def find_boundaries(self, similarity_scores: list, std_dev_factor: float) -> list:
        """Find topic boundaries as valleys below dynamic threshold."""
        mean_score = np.mean(similarity_scores)
        std_dev_score = np.std(similarity_scores)
        dynamic_threshold = mean_score - (std_dev_factor * std_dev_score)
        print(f"   - Mean Similarity: {mean_score:.4f}")
        print(f"   - Std Deviation:   {std_dev_score:.4f}")
        print(f"   - DYNAMIC THRESHOLD: {dynamic_threshold:.4f}")

        print("⏳ Finding potential topic boundaries (valleys)...")
        valley_indices = argrelextrema(np.array(similarity_scores), np.less)[0]

        print("⏳ Filtering valleys based on the dynamic threshold...")
        return [i for i in valley_indices if similarity_scores[i] < dynamic_threshold]




# import numpy as np
# from scipy.signal import argrelextrema
# import nltk
# from typing import List, Union

# class NLPUtils:
#     def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
#         """Calculates the cosine similarity between two numpy vectors."""
#         dot_product = np.dot(v1, v2)
#         norm_v1 = np.linalg.norm(v1)
#         norm_v2 = np.linalg.norm(v2)
#         return dot_product / (norm_v1 * norm_v2)

#     def sentence_tokenize(self, transcript_data: Union[str, list]) -> list: #text: str) -> list:
#         try:
#             nltk.data.find('tokenizers/punkt')
#         except LookupError:
#             print("⏳ First-time setup: Downloading NLTK's 'punkt' model for sentence splitting...")
#             nltk.download('punkt')
#             try:
#                 nltk.data.find('tokenizers/punkt_tab')
#             except LookupError:
#                 print("⏳ First-time setup: Downloading NLTK's 'punkt_tab' dependency...")
#                 nltk.download('punkt_tab')
#             print("✅ NLTK setup complete.")

#         # Handle different input formats
#         if isinstance(transcript_data, list):
#             # Assume it's a list of Whisper segments
#             text = " ".join([segment.get("text", "") for segment in transcript_data])
#         elif isinstance(transcript_data, str):
#             # Check if it's a JSON string
#             try:
#                 json_data = json.loads(transcript_data)
#                 if isinstance(json_data, list):
#                     text = " ".join([segment.get("text", "") for segment in json_data])
#                 else:
#                     text = transcript_data
#             except json.JSONDecodeError:
#                 # It's just plain text
#                 text = transcript_data
#         else:
#             raise ValueError("Unsupported transcript data format")
        
#         return nltk.sent_tokenize(text)

#     def find_boundaries(self, similarity_scores: list, std_dev_factor: float) -> list:
#         mean_score = np.mean(similarity_scores)
#         std_dev_score = np.std(similarity_scores)
#         dynamic_threshold = mean_score - (std_dev_factor * std_dev_score)
#         print(f"   - Mean Similarity: {mean_score:.4f}")
#         print(f"   - Std Deviation:   {std_dev_score:.4f}")
#         print(f"   - DYNAMIC THRESHOLD: {dynamic_threshold:.4f}")

#         print("⏳ Finding potential topic boundaries (valleys)...")
#         valley_indices = argrelextrema(np.array(similarity_scores), np.less)[0]

#         print("⏳ Filtering valleys based on the dynamic threshold...")
#         return [i for i in valley_indices if similarity_scores[i] < dynamic_threshold]