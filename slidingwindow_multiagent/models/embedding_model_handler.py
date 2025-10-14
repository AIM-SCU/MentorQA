import sys
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from common_utils.paths import bge_model_path

class EmbedModel:
    def __init__(self, model_path = bge_model_path):
        self.embed_model = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"batch_size": 16}
            )

    def embed_documents(self, documents: list) -> list:
        return self.embed_model.embed_documents(documents)

    def cleanup(self):
        del self.embed_model
        torch.cuda.empty_cache()