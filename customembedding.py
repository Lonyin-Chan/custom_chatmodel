from typing import List

from langchain_core.embeddings import Embeddings

from langchain_ollama import OllamaEmbeddings


class CustomEmbeddings(Embeddings):

    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""

        embeddings = OllamaEmbeddings(
            model=self.model
        )

        vectors = embeddings.embed_documents(texts)

        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

