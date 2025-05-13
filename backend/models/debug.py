"""
Debug Helper Script for NEPSE-Navigator Vector DB Verification

Updated version with:
- Accurate docstore file detection (checks for .pkl files)
- Consistent OS path handling
- Clearer diagnostic messages
- Suppressed non-critical FAISS warnings
"""

import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress non-critical FAISS warnings
logging.getLogger('faiss').setLevel(logging.ERROR)

# Path to data directory - using consistent os.path.join()
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vector_db_paths = {
    # 'broker': os.path.join(base_dir, "data", "vector data", "broker_vector_db"),
    # 'fundamental': os.path.join(base_dir, "data", "vector data", "fundamental_vector_db"),
    # 'company': os.path.join(base_dir, "data", "vector data", "company_vector_db"),
    # 'pdf': os.path.join(base_dir, "data", "vector data", "pdf_vector_db")
    'data': os.path.join(base_dir, "data", "vector data", "data_vector_db")
}

def check_vector_dbs():
    """Verify each vector DB and report status"""
    print("\n" + "="*50)
    print("NEPSE-Navigator Vector Database Diagnostic Tool")
    print("="*50)
    
    # Check directory structure and files
    for name, path in vector_db_paths.items():
        print(f"\nChecking {name} vector database:")
        print(f"  Path: {path}")
        
        # Directory check
        if not os.path.exists(path):
            print(f"  ❌ ERROR: Directory does not exist")
            print(f"  SOLUTION: Create directory or check path in config")
            continue
            
        print(f"  ✅ Directory exists")
        
        # File checks
        required_files = ['index.faiss', 'index.pkl']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(path, f))]
        
        if missing_files:
            print(f"  ❌ ERROR: Missing required files: {', '.join(missing_files)}")
            print(f"  SOLUTION: Rebuild vector store or check creation process")
            continue
            
        print(f"  ✅ Found all required files: {', '.join(required_files)}")
        
    # Try to load and query
    print("\nAttempting to load and query vector DBs...")
    try:
        # Load embeddings
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        print(f"\nLoading embeddings model: {model_name}")
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
        )
        print("✅ Embeddings model loaded successfully")
        
        # Test each DB
        for name, path in vector_db_paths.items():
            if not os.path.exists(path):
                continue
                
            print(f"\nTesting {name} vector DB...")
            try:
                # Load DB
                db = FAISS.load_local(
                    path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                doc_count = len(db.index_to_docstore_id)
                print(f"✅ Successfully loaded ({doc_count} documents)")
                
                # Test query
                query = "what is nepse?"
                print(f"  Testing query: '{query}'")
                results = db.similarity_search_with_score(query, k=1)
                
                if results:
                    doc, score = results[0]
                    print(f"  ✅ Found result (score: {score:.2f})")
                    print(f"  Sample content: {doc.page_content[:100]}...")
                else:
                    print("  ❌ No results found (vector store may be empty)")
                    
            except Exception as e:
                print(f"❌ Failed to load/query: {str(e)}")
                print("  SOLUTION: Check if DB was created with same embeddings model")
    
    except Exception as e:
        print(f"\n❌ Critical error during testing: {str(e)}")
    
    # Summary
    print("\n" + "="*50)
    print("DIAGNOSTIC SUMMARY:")
    print("✅ Working correctly if:")
    print("  - All required files exist (index.faiss, index.pkl)")
    print("  - DB loads without errors")
    print("  - Queries return results")
    print("\n⚠️ Common issues:")
    print("  - Missing files → Rebuild vector stores")
    print("  - Loading errors → Check embeddings model consistency")
    print("  - Empty results → Verify source data was properly indexed")
    print("="*50)

if __name__ == "__main__":
    check_vector_dbs()