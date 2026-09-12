import os
import time
from loguru import logger
from langchain_community.vectorstores import FAISS
from utils.utils import get_all_files
from step_one.document_processor import DocumentProcessor
from langchain_huggingface import HuggingFaceEmbeddings

import torch
torch.set_num_threads(8)


class VectorStoreManager:
    """Implement the vector store manager component."""
    def __init__(self, embedding_model, cache_dir, store_path, doc_folder):
        """Initialize the instance."""
        start_time = time.time()
        self.embeddings = HuggingFaceEmbeddings(
                                        model_name=cache_dir + '/' + embedding_model,
                                        model_kwargs={'device': 'cpu'}
                                    )

        self.doc_folder = doc_folder
        self.store_path = store_path
        self.vector_store = None
        end_time = time.time()
        logger.info(f"Embedding model initialization time: {end_time - start_time:.4f} seconds")


    def create_embedding_vector(self, doc_folder):
        """Create embedding vector."""
        all_chunks = []
        yaml_files = get_all_files(doc_folder, '.yaml')

        for yaml_file in yaml_files:
            api_name, *_ = DocumentProcessor().doc2str(file_path=yaml_file)
            all_chunks.append(api_name)

        self.vector_store = FAISS.from_texts(
            texts=all_chunks,
            embedding=self.embeddings
        )
        self.vector_store.save_local(self.store_path)
        logger.info("Vector store created")

    def load_embedding_vector(self):
        """Load embedding vector."""
        self.vector_store = FAISS.load_local(
            folder_path=self.store_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

    def main(self, query, k):
        """Run the main workflow."""
        if self.vector_store is None:
            if os.path.exists(self.store_path):
                logger.info(f"Vector store {self.store_path} exists; loading it")
                self.load_embedding_vector()
            else:
                logger.info(f"Vector store {self.store_path} does not exist; creating it first")
                self.create_embedding_vector(self.doc_folder)
                self.load_embedding_vector()

        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        doc = [doc[0].page_content for doc in results]

        return doc
