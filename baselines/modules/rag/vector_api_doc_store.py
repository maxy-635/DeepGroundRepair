import os
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from baselines.modules.rag.document_processor import APIDocProcessor
from utils.utils import get_all_files


class VectorAPIDocStore:
    """Dense retriever that embeds complete api_doc strings."""

    def __init__(self, embedding_model: str, cache_dir: str, store_path: str, doc_folder: str):
        start_time = time.time()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.path.join(cache_dir, embedding_model),
            model_kwargs={"device": "cpu"},
        )
        self.store_path = store_path
        self.doc_folder = doc_folder
        self.vector_store = None
        logger.info(f"Embedding model initialization time: {time.time() - start_time:.4f} seconds")

    def create_embedding_vector(self) -> None:
        processor = APIDocProcessor()
        texts = []
        metadatas = []

        for yaml_file in get_all_files(self.doc_folder, ".yaml"):
            api_doc = processor.yaml2api_doc(yaml_file)
            texts.append(api_doc["api_doc"])
            metadatas.append({"api_name": api_doc["api_name"]})

        logger.info(f"Documents queued for the api_doc vector store: {len(texts)}")
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
        )
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        self.vector_store.save_local(self.store_path)
        logger.info(f"api_doc vector store saved to {self.store_path}")

    def load_embedding_vector(self) -> None:
        self.vector_store = FAISS.load_local(
            folder_path=self.store_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def main(self, query: str, k: int) -> list[dict]:
        if self.vector_store is None:
            if os.path.exists(self.store_path):
                logger.info(f"api_doc vector store {self.store_path} exists; loading it.")
                self.load_embedding_vector()
            else:
                logger.info(f"api_doc vector store {self.store_path} does not exist; creating it first.")
                self.create_embedding_vector()
                self.load_embedding_vector()

        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)

        retrieved_results = [
            {
                "api_name": doc.metadata.get("api_name", ""),
                "api_doc": doc.page_content,
                "score": round(score, 4),
            }
            for doc, score in results
        ]

        return retrieved_results
