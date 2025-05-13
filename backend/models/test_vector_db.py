# test_vector_db.py
from chat import PredictionPipeline
import sys

pipeline = PredictionPipeline()
pipeline.load_sentence_transformer()
pipeline.load_embeddings()

def test_query(query):
    print(f"\nTesting query: '{query}'")
    docs = pipeline.vector_db.similarity_search_with_score(query, k=5)
    for i, (doc, score) in enumerate(docs):
        print(f"\nDocument {i+1} (Score: {score:.3f}):")
        print(doc.page_content[:500] + "...")

if __name__ == "__main__":
    test_query("minimum capital requirement for broker company")
    test_query("IPO process in Nepal")
    test_query("highest trading volume stock")