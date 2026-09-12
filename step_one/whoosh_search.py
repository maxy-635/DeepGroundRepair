import os
import jieba
from loguru import logger
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
from whoosh.analysis import Token, Analyzer
from whoosh.scoring import BM25F
from step_one.document_processor import DocumentProcessor
from utils.utils import get_all_files


class JiebaAnalyzer(Analyzer):
    """Implement the jieba analyzer component."""
    def __call__(self, value, positions=True, chars=True,
                 keeporiginal=False, removestops=True, start_pos=0,
                 start_char=0, mode='', **kwargs):
        t = Token(positions, chars, removestops=removestops,
                  mode=mode, **kwargs)

        pos = start_pos
        char_pos = start_char
        for word in jieba.cut(value):
            word = word.strip()
            if not word:
                continue
            t.original = t.text = word
            if positions:
                t.pos = pos
                pos += 1
            if chars:
                t.startchar = char_pos
                t.endchar = char_pos + len(word)
                char_pos = t.endchar
            yield t

class WhooshSearch(JiebaAnalyzer):
    """Implement the whoosh search component."""
    def __init__(self, docs_path, index_dir):
        """Initialize the instance."""
        self.docs_path = docs_path
        self.index_dir = index_dir
        
        self.schema = Schema(
            api_name = TEXT(stored=True, analyzer=JiebaAnalyzer()),
            api_description = TEXT(stored=True, analyzer=JiebaAnalyzer()),
            api_signature = TEXT(stored=True, analyzer=JiebaAnalyzer()),
            api_details = TEXT(stored=True, analyzer=JiebaAnalyzer()),
            api_usage_description = TEXT(stored=True, analyzer=JiebaAnalyzer()),
            api_parameters = TEXT(stored=True, analyzer=JiebaAnalyzer()),
            api_usage_example = TEXT(stored=True, analyzer=JiebaAnalyzer()),
            api_shape_formula = TEXT(stored=True, analyzer=JiebaAnalyzer()),
        )

        if not os.path.exists(self.index_dir):
            os.mkdir(self.index_dir)
            self.ix = create_in(self.index_dir, self.schema)
            self.index_created = False
            logger.info(f"Index directory {self.index_dir} does not exist; created the index.")
        else:
            self.ix = open_dir(self.index_dir)
            self.index_created = True
            logger.info(f"Index directory {self.index_dir} exists; opening the index.")

    def add_documents(self):
        """Add documents."""
        writer = self.ix.writer()

        for yaml_file in get_all_files(self.docs_path, '.yaml'):
            api_name, api_description, api_signatures, api_details, api_usage_description, api_parameters, api_shape_formula, api_usage_example = DocumentProcessor().doc2str(yaml_file)
            writer.add_document(
                api_name=api_name,
                api_description=api_description,
                api_signature=api_signatures,
                api_details=api_details,
                api_usage_description=api_usage_description,
                api_parameters=api_parameters,
                api_usage_example=api_usage_example,
                api_shape_formula=api_shape_formula,
            )

        writer.commit()


    def search(self, query_str, limit):

        def format_result(data, score):
            """Format result."""
            return {
                "api_name": data["api_name"],
                "api_description": data["api_description"],
                "api_signature": data["api_signature"],
                "api_details": data["api_details"],
                "api_usage_description": data["api_usage_description"],
                "api_parameters": data["api_parameters"],
                "api_usage_example": data["api_usage_example"],
                "api_shape_formula": data["api_shape_formula"],
                "score": round(score, 4)
            }

        with self.ix.searcher(weighting=BM25F(B=0.75, K1=1.2)) as searcher:
            parser = QueryParser("api_name", schema=self.schema)
            query = parser.parse(query_str)

            results = searcher.search(query, limit=limit, scored=True)
            if results:
                output = [format_result(result, result.score) for result in results]

            else:
                logger.info(f"BM25 exact retrieval returned no matches")
                output = []

            return output

    
    def main(self, query, limit):
        """Run the main workflow."""
        if not self.index_created:
            logger.info("Adding documents to the index")
            self.add_documents()
            results = self.search(query, limit)
            self.index_created = True
        else:
            results = self.search(query, limit)
        
        return results
