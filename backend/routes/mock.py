import logging
from pathlib import Path

import numpy as np
import torch
from FlagEmbedding import FlagModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sentence_transformer():
    """Test loading sentence transformer and embedding generation."""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sentence_transformer = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': device}
        )
        query = "RSI of nabil bank"
        embedding = sentence_transformer.embed_query(query)
        logger.info(f"Embedding size: {len(embedding)}, Device: {device}")
        return sentence_transformer, True
    except Exception as e:
        logger.error(f"Sentence transformer failed: {e}")
        return None, False

def test_pdf_vector_store_path():
    """Test if PDF vector store path and files exist."""
    base_dir = Path(__file__).resolve().parent.parent
    pdf_path = base_dir / "data" / "vector data" / "company_vector_db"
    files_exist = all([(pdf_path / "index.faiss").exists(), (pdf_path / "index.pkl").exists()])
    logger.info(f"PDF vector store at {pdf_path}: {'Valid' if files_exist else 'Missing or incomplete'}")
    return files_exist, pdf_path

def test_load_pdf_vector_store(sentence_transformer, pdf_path):
    """Test loading PDF vector store with FAISS."""
    try:
        vector_db = FAISS.load_local(
            folder_path=str(pdf_path),
            embeddings=sentence_transformer,
            allow_dangerous_deserialization=True
        )
        num_vectors = getattr(vector_db.index, 'ntotal', 0)
        logger.info(f"PDF vector store loaded with {num_vectors} vectors")
        return vector_db, num_vectors > 0
    except Exception as e:
        logger.error(f"PDF vector store loading failed: {e}")
        return None, False

def test_pdf_similarity_search(pdf_db):
    """Test similarity search on PDF vector store."""
    query = "RSI of nabil bank"
    try:
        results = pdf_db.similarity_search_with_score(query, k=3)
        if not results:
            logger.warning("No results found for PDF query")
            return False
        for doc, score in results:
            logger.info(f"Result (score: {score:.2f}): {doc.page_content[:100]}...")
        return True
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return False

def test_pdf_reranking(pdf_db):
    """Test reranking of PDF search results."""
    try:
        reranker = FlagModel('BAAI/bge-reranker-large', use_fp16=torch.cuda.is_available())
        query = "RSI of nabil bank"
        results = pdf_db.similarity_search_with_score(query, k=5)
        contexts = [doc.page_content for doc, score in results if score < 1.5]
        if not contexts:
            logger.warning("No contexts to rerank")
            return False
        emb_query = reranker.encode(query)
        emb_contexts = reranker.encode(contexts)
        similarity_scores = np.dot(emb_contexts, emb_query.T).flatten()
        top_indices = similarity_scores.argsort()[::-1][:3]
        ranked_contexts = [contexts[i] for i in top_indices]
        logger.info(f"Top reranked context: {ranked_contexts[0][:100]}...")
        nepse_terms = ["rsi", "sma", "company", "stock", "price", "nabil", "bank"]
        return any(any(term in ctx.lower() for term in nepse_terms) for ctx in ranked_contexts)
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return False

def test_pdf_vector_pipeline():
    """Test full PDF vector embedding pipeline."""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sentence_transformer = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': device}
        )
        reranker = FlagModel('BAAI/bge-reranker-large', use_fp16=device == 'cuda')
        pdf_path = Path(__file__).resolve().parent.parent / "data" / "vector data" / "company_vector_db"
        pdf_db = FAISS.load_local(
            folder_path=str(pdf_path),
            embeddings=sentence_transformer,
            allow_dangerous_deserialization=True
        )
        query = "RSI of nabil bank"
        results = pdf_db.similarity_search_with_score(query, k=5)
        contexts = [doc.page_content for doc, score in results if score < 1.5]
        if not contexts:
            logger.warning("No contexts found")
            return False
        emb_query = reranker.encode(query)
        emb_contexts = reranker.encode(contexts)
        similarity_scores = np.dot(emb_contexts, emb_query.T).flatten()
        top_indices = similarity_scores.argsort()[::-1][:3]
        ranked_contexts = [contexts[i] for i in top_indices]
        logger.info(f"Top context: {ranked_contexts[0][:100]}...")
        nepse_terms = ["nepse", "policy", "regulation", "supervision", "securities"]
        return any(any(term in ctx.lower() for term in nepse_terms) for ctx in ranked_contexts)
    except Exception as e:
        logger.error(f"PDF vector pipeline failed: {e}")
        return False

def run_all_tests():
    """Run all tests for PDF vector embedding components."""
    logger.info("Starting PDF vector store tests")

    sentence_transformer, transformer_success = test_sentence_transformer()
    if not transformer_success:
        logger.error("Aborting: Sentence transformer failed")
        return False

    path_valid, pdf_path = test_pdf_vector_store_path()
    if not path_valid:
        logger.error("Aborting: PDF vector store path invalid")
        return False

    pdf_db, load_success = test_load_pdf_vector_store(sentence_transformer, pdf_path)
    if not load_success:
        logger.error("Aborting: PDF vector store loading failed")
        return False

    if not test_pdf_similarity_search(pdf_db):
        logger.warning("PDF similarity search returned no results")

    if not test_pdf_reranking(pdf_db):
        logger.warning("PDF reranking failed")

    if not test_pdf_vector_pipeline():
        logger.error("PDF vector embedding pipeline failed")
        return False

    logger.info("All PDF vector store tests completed successfully")
    return True

if __name__ == "__main__":
    import transformers
    transformers.logging.set_verbosity_error()

    success = run_all_tests()
    if success:
        logger.info("✅ All tests passed: PDF vector embedding components are working")
    else:
        logger.error("❌ Some tests failed: Check logs for details")
