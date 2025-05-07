from typing import Any, Dict, List

import numpy as np
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from transformers import AutoTokenizer

# ---- Load TinyLlama GPTQ Model ----
model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(
    model_id,
    use_safetensors=True,
    trust_remote_code=True,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# ---- Load FAISS vector stores ----
from langchain_community.vectorstores import FAISS

# Vector DB configurations with metadata
vector_dbs = {
    "fundamental": {
        "db": FAISS.load_local("../data/vector data/fundamental_vector_db", embeddings=embeddings, allow_dangerous_deserialization=True),
        "weight": 1.0  # Default weight
    },
    "company": {
        "db": FAISS.load_local("../data/vector data/company_vector_db", embeddings=embeddings, allow_dangerous_deserialization=True),
        "weight": 1.0
    },
    "broker": {
        "db": FAISS.load_local("../data/vector data/broker_vector_db", embeddings=embeddings, allow_dangerous_deserialization=True),
        "weight": 1.0
    },
    "pdf": {
        "db": FAISS.load_local("../data/vector data/pdf_vector_db", embeddings=embeddings, allow_dangerous_deserialization=True),
        "weight": 1.0
    }
}

# ---- Detect Language ----
def detect_language(text):
    try:
        lang = detect(text)
        if lang in ["en", "ne"]:
            return lang
        else:
            return "unsupported"
    except LangDetectException:
        return "unsupported"

# ---- Improved Vector DB Search with Re-ranking ----
def search_vector_dbs(query, k=3, topic_weights=None):
    """
    Search across all vector DBs with optional topic weighting and re-ranking
    
    Args:
        query: The search query
        k: Number of results per DB
        topic_weights: Optional dict to override default weights based on query topic
    
    Returns:
        List of documents sorted by relevance
    """
    # Apply topic-specific weights if provided
    weights = {}
    for db_name in vector_dbs:
        weights[db_name] = topic_weights.get(db_name, vector_dbs[db_name]["weight"]) if topic_weights else vector_dbs[db_name]["weight"]
    
    # Get results from each DB with scores
    all_results = []
    for db_name, db_config in vector_dbs.items():
        try:
            # Get results with scores
            docs_with_scores = db_config["db"].similarity_search_with_score(query, k=k)
            
            # Apply DB-specific weight to scores
            weighted_results = [(doc, score * weights[db_name], db_name) for doc, score in docs_with_scores]
            all_results.extend(weighted_results)
        except Exception as e:
            print(f"Error searching {db_name} DB: {e}")
    
    # Sort by weighted score (lower is better with FAISS distances)
    all_results.sort(key=lambda x: x[1])
    
    # Take top results and return only the documents
    top_docs = [item[0] for item in all_results[:k*2]]  # Getting top k*2 docs across all DBs
    
    return top_docs

# ---- Query Classification for Adaptive Retrieval ----
def classify_query_topic(query):
    """
    Simple keyword-based query classifier to determine weights for different DBs
    """
    query = query.lower()
    
    # Define keywords for each category
    keywords = {
        "fundamental": ["stock", "market", "index", "trend", "fundamental", "analysis", "ratio", "pe", "eps"],
        "company": ["company", "organization", "business", "financial statement", "balance sheet", "profit", "revenue"],
        "broker": ["broker", "trading", "commission", "fee", "account", "trade"],
        "pdf": ["report", "document", "research", "analysis", "pdf"]
    }
    
    # Count keyword matches
    counts = {db_name: 0 for db_name in vector_dbs}
    for db_name, db_keywords in keywords.items():
        for keyword in db_keywords:
            if keyword in query:
                counts[db_name] += 1
    
    # Calculate weights - if no matches, use equal weights
    if sum(counts.values()) > 0:
        weights = {db_name: 1.0 + (count * 0.5) for db_name, count in counts.items()}
    else:
        weights = {db_name: 1.0 for db_name in vector_dbs}
    
    return weights

# ---- Enhanced Prompt Template ----
def build_prompt(query, context_docs, lang="en"):
    # Extract source information from metadata if available
    context_parts = []
    for i, doc in enumerate(context_docs):
        # Get source info from metadata if available
        source_info = ""
        if hasattr(doc, 'metadata') and doc.metadata:
            if 'source' in doc.metadata:
                source_info = f" (Source: {doc.metadata['source']})"
        
        # Add document with source info
        context_parts.append(f"Document {i+1}{source_info}:\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)
    
    if lang == "ne":
        return f"""### निर्देशन:
तल दिइएको सन्दर्भ प्रयोग गरेर प्रयोगकर्ताको प्रश्नको उत्तर दिनुहोस्। यदि सन्दर्भमा जानकारी छैन भने, तपाईंलाई थाहा छैन भनेर स्पष्ट रूपमा भन्नुहोस्। सन्दर्भमा दिईएको जानकारी मात्र प्रयोग गर्नुहोस् र आफू अनुमान नगर्नुहोस्।

### सन्दर्भ:
{context}

### प्रश्न:
{query}

### उत्तर:"""
    else:
        return f"""### Instruction:
Use the following context to answer the user's question clearly and informatively. If the information is not in the context, clearly state that you don't know. Rely strictly on the provided context and avoid making assumptions.

### Context:
{context}

### Question:
{query}

### Answer:"""

# ---- Response Generation with Error Handling ----
def get_rag_response(query, max_tokens=300):
    try:
        # Detect language
        lang = detect_language(query)
        if lang == "unsupported":
            return "❌ Please ask your question in either English or Nepali only."
        
        # Classify query to determine DB weights
        topic_weights = classify_query_topic(query)
        
        # Retrieve relevant documents
        retrieved_docs = search_vector_dbs(query, k=3, topic_weights=topic_weights)
        if not retrieved_docs:
            return "❌ No relevant information found to answer your query."
        
        # Build prompt with retrieved context
        prompt = build_prompt(query, retrieved_docs, lang)
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2
            )
        
        # Extract answer from response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("### Answer:")[-1].strip()
        
        return answer
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error generating response: {error_details}")
        return f"❌ Sorry, an error occurred while processing your query. Please try again or rephrase your question."

# ---- Enhanced Example Usage ----
if __name__ == "__main__":
    print("💬 RAG System with TinyLlama (1.1B) - Support for English and Nepali")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("\n📝 Enter your query (English or Nepali): ")
        if query.lower() in ['exit', 'quit']:
            break
            
        print("\n⏳ Processing...")
        answer = get_rag_response(query)
        print("\n🧠 Answer:\n", answer)
