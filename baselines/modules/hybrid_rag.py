from collections import defaultdict
from loguru import logger
from baselines.modules.rag.whoosh_api_doc_search import WhooshAPIDocSearch
from baselines.modules.rag.vector_api_doc_store import VectorAPIDocStore



class HybridRAGRetriever:
    """Implement the hybrid ragretriever component."""

    def __init__(self, embedding_model_dir: str, dll: str, top_k: int) -> None:
        """Initialize the instance."""

        config = {
            "whoosh": {"docs_path": f"./api_parser/{dll.lower()}/postprocessed_apis_v2", 
                       "index_dir": f"./baselines/B2_databases/whoosh/whoosh_{dll.lower()}"},
            "embedding": {"embedding_model": "models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
                        "cache_dir": embedding_model_dir,
                        "store_path": f"./baselines/B2_databases/embedding/embedding_{dll.lower()}.faiss",
                        "doc_folder": f"./api_parser/{dll.lower()}/postprocessed_apis_v2"}
            }

        self.whoosh_searcher = WhooshAPIDocSearch(**config["whoosh"])
        self.embedding_searcher = VectorAPIDocStore(**config["embedding"])
        self.top_k = top_k


    def retrieve(self, query: str) -> list[dict]:

        # Step1: Sparse retrieval
        sparse_results = self.whoosh_searcher.main(
            query=query,
            limit=self.top_k,
        )
        logger.info("Whoosh sparse results:")
        for idx, doc in enumerate(sparse_results, start=1):
            logger.info(f"{idx}. {doc.get('api_name', '')}")

        # Step2: Dense retrieval
        dense_results = self.embedding_searcher.main(
            query=query,
            k=self.top_k,
        )
        if not dense_results:
            logger.warning("Embedding dense retrieval returned no results.")
        logger.info("Embedding dense results:")
        for idx, doc in enumerate(dense_results, start=1):
            logger.info(f"{idx}. {doc.get('api_name', '')}")

        # Step3: Reciprocal Rank Fusion (RRF)
        scores = defaultdict(float)
        docs_by_name = {}

        for ranked_list in [sparse_results, dense_results]:
            for rank, doc in enumerate(ranked_list, start=1):
                api_name = doc.get("api_name", "")
                scores[api_name] += 1.0 / (60 + rank)
                docs_by_name.setdefault(api_name, doc)

        fused_results = []
        fused_names = sorted(scores,key=scores.get,reverse=True)
        for api_name in fused_names[: self.top_k]:
            doc = dict(docs_by_name[api_name])
            doc["fusion_score"] = round(scores[api_name], 6)
            fused_results.append(doc)

        logger.info("Fused results:")
        for idx, doc in enumerate(fused_results, start=1):
            logger.info(f"{idx}. {doc.get('api_name', '')}, Fusion Score: {doc['fusion_score']}")

        return fused_results
