import os
import jieba
from loguru import logger
from whoosh.analysis import Analyzer, Token
from whoosh.fields import ID, Schema, TEXT
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from whoosh.scoring import BM25F
from baselines.modules.rag.document_processor import APIDocProcessor
from utils.utils import get_all_files


class JiebaAnalyzer(Analyzer):
    """Use Jieba as the tokenizer for Whoosh."""

    def __call__(self, value, positions=True, chars=True,
                 keeporiginal=False, removestops=True, start_pos=0,
                 start_char=0, mode='', **kwargs):
        token = Token(positions, chars, removestops=removestops, mode=mode, **kwargs)
        pos = start_pos
        char_pos = start_char
        for word in jieba.cut(value):
            word = word.strip()
            if not word:
                continue
            token.original = token.text = word
            if positions:
                token.pos = pos
                pos += 1
            if chars:
                token.startchar = char_pos
                token.endchar = char_pos + len(word)
                char_pos = token.endchar
            yield token


class WhooshAPIDocSearch:
    """Sparse retriever whose only searchable field is the complete api_doc."""

    def __init__(self, docs_path: str, index_dir: str):
        self.docs_path = docs_path
        self.index_dir = index_dir
        self.schema = Schema(
            api_name=ID(stored=True),
            api_doc=TEXT(stored=True, analyzer=JiebaAnalyzer()),
        )

        os.makedirs(os.path.dirname(self.index_dir), exist_ok=True)
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir, exist_ok=True)
            self.ix = create_in(self.index_dir, self.schema)
            self.index_created = False
            logger.info(f"Whoosh api_doc index {self.index_dir} does not exist; created an empty index.")
        else:
            self.ix = open_dir(self.index_dir)
            self.index_created = True
            logger.info(f"Whoosh api_doc index {self.index_dir} exists; opening it.")

    def add_documents(self) -> None:
        writer = self.ix.writer()
        processor = APIDocProcessor()

        for yaml_file in get_all_files(self.docs_path, ".yaml"):
            api_doc = processor.yaml2api_doc(yaml_file)
            writer.add_document(
                api_name=api_doc["api_name"],
                api_doc=api_doc["api_doc"],
            )

        writer.commit()
        logger.info("Whoosh api_doc index built.")

    def search(self, query: str, limit: int) -> list[dict]:
        with self.ix.searcher(weighting=BM25F(B=0.75, K1=1.2)) as searcher:
            parser = QueryParser("api_doc", schema=self.schema)
            parsed_query = parser.parse(query)
            results = searcher.search(parsed_query, limit=limit, scored=True)

            return [
                {
                    "api_name": result["api_name"],
                    "api_doc": result["api_doc"],
                    "score": round(result.score, 4),
                }
                for result in results
            ]

    def main(self, query: str, limit: int) -> list[dict]:
        if not self.index_created:
            self.add_documents()
            self.index_created = True

        retrieved_results = self.search(query, limit)

        return retrieved_results
