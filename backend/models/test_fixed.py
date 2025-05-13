import os
import sys
import torch
import psutil
import traceback
import time
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import

class NepseGPTDiagnostics:
    def __init__(self, *vector_store_paths):
        self.vector_store_paths = vector_store_paths
        self.faiss_stores = {}
        print("Initializing NEPSE-GPT Diagnostics...")
        self.log_system_info()
        
    def log_system_info(self):
        """Log system information"""
        print("==== System Information ====")
        print(f"Python version: {sys.version}")
        print(f"CPU Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        
        if torch.cuda.is_available():
            print(f"CUDA available: Yes")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
        else:
            print("CUDA available: No")
        print("============================")
    
    def test_cuda_extension(self):
        """Test CUDA extension and basic tensor operations"""
        print("\n==== Testing CUDA ====")
        try:
            # Force CUDA initialization
            if torch.cuda.is_available():
                device = torch.device("cuda")
                # Create a small tensor to initialize CUDA
                dummy = torch.ones(1).to(device)
                # Try a simple operation
                result = dummy + dummy
                print(f"CUDA tensor operation successful: {result.item() == 2}")
                print(f"CUDA initialized successfully on {torch.cuda.get_device_name(0)}")
            else:
                print("CUDA not available")
        except Exception as e:
            print(f"CUDA initialization error: {str(e)}")
            print(traceback.format_exc())
        print("=====================")
    
    def load_embeddings(self):
        """Load and test sentence transformer embeddings"""
        print("\n==== Testing Embeddings ====")
        try:
            start_time = time.time()
            # Explicitly specify a model name to avoid deprecation warning
            self.sentence_transformer = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            load_time = time.time() - start_time
            print(f"Sentence transformer loaded in {load_time:.2f} seconds")
            
            # Test embedding generation
            test_text = "This is a test sentence for embedding"
            start_time = time.time()
            embedding = self.sentence_transformer.embed_query(test_text)
            embed_time = time.time() - start_time
            
            print(f"Generated embedding of dimension: {len(embedding)}")
            print(f"Embedding generation time: {embed_time:.4f} seconds")
            print(f"Embedding stats - min: {min(embedding):.4f}, max: {max(embedding):.4f}")
            print(f"Memory usage after embedding:")
            self.log_memory_usage()
            return True
        except Exception as e:
            print(f"Embedding error: {str(e)}")
            print(traceback.format_exc())
            return False
        finally:
            print("=========================")
    
    def list_directory_structure(self, path):
        """List directory structure of a path"""
        print(f"\n==== Directory Structure for {path} ====")
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                print(f"{file}/")
            else:
                print(f"{file}")
        print("===========================")
    
    def inspect_vector_stores(self):
        """Inspect all FAISS vector stores in the given paths"""
        print("\n==== Inspecting Vector Stores ====")
        
        if not self.vector_store_paths:
            print("No vector store paths provided!")
            return False
        
        print(f"Examining {len(self.vector_store_paths)} vector store paths:")
        for i, path in enumerate(self.vector_store_paths):
            print(f"{i+1}. {path}")
            self.list_directory_structure(path)
        
        # Try to load each vector store with allow_dangerous_deserialization=True
        success_count = 0
        for i, path in enumerate(self.vector_store_paths):
            store_name = os.path.basename(path)
            print(f"\nInspecting vector store {i+1}: {store_name}")
            try:
                start_time = time.time()
                # IMPORTANT: Set allow_dangerous_deserialization=True
                vector_store = FAISS.load_local(
                    path, 
                    self.sentence_transformer,
                    allow_dangerous_deserialization=True  # This is the key fix
                )
                load_time = time.time() - start_time
                
                # Store for later use
                self.faiss_stores[store_name] = vector_store
                
                # Get statistics
                index_size = vector_store.index.ntotal if hasattr(vector_store, 'index') else "Unknown"
                print(f"✓ Successfully loaded in {load_time:.2f} seconds")
                print(f"- Index size: {index_size} vectors")
                
                # Check docstore
                if hasattr(vector_store, 'docstore') and hasattr(vector_store.docstore, '_dict'):
                    doc_count = len(vector_store.docstore._dict)
                    print(f"- Document count: {doc_count}")
                    if doc_count > 0:
                        # Print sample doc
                        sample_key = next(iter(vector_store.docstore._dict))
                        sample_doc = vector_store.docstore._dict[sample_key]
                        print(f"- Sample document ID: {sample_key}")
                        print(f"- Sample document page content length: {len(sample_doc.page_content) if hasattr(sample_doc, 'page_content') else 'Unknown'}")
                        print(f"- Sample document metadata: {sample_doc.metadata if hasattr(sample_doc, 'metadata') else 'None'}")
                
                success_count += 1
            except Exception as e:
                print(f"✗ Failed to load: {str(e)}")
                print(traceback.format_exc())
        
        print(f"\nSuccessfully loaded {success_count} out of {len(self.vector_store_paths)} vector stores")
        print("===============================")
        return success_count > 0
    
    def test_similarity_search(self, query="What is NEPSE index?"):
        """Test similarity search on all loaded vector stores"""
        print(f"\n==== Testing Similarity Search with '{query}' ====")
        if not self.faiss_stores:
            print("No vector stores loaded. Run inspect_vector_stores() first.")
            return False
        
        success_count = 0
        for store_name, vector_store in self.faiss_stores.items():
            print(f"\nTesting search in vector store: {store_name}")
            try:
                # First try with query string
                start_time = time.time()
                results = vector_store.similarity_search(query, k=3)
                search_time = time.time() - start_time
                
                print(f"✓ String search successful in {search_time:.4f} seconds")
                print(f"- Found {len(results)} documents")
                
                if results:
                    print(f"- Top result content preview: {results[0].page_content[:100]}...")
                    print(f"- Top result metadata: {results[0].metadata}")
                
                # Now try with embedding vector
                query_embedding = self.sentence_transformer.embed_query(query)
                start_time = time.time()
                results = vector_store.similarity_search_by_vector(query_embedding, k=3)
                vector_search_time = time.time() - start_time
                
                print(f"✓ Vector search successful in {vector_search_time:.4f} seconds")
                print(f"- Found {len(results)} documents")
                
                success_count += 1
            except Exception as e:
                print(f"✗ Search failed: {str(e)}")
                print(traceback.format_exc())
        
        print(f"\nSuccessfully searched {success_count} out of {len(self.faiss_stores)} vector stores")
        print("===============================================")
        return success_count > 0
    
    def test_all_queries(self):
        """Test all benchmark queries"""
        test_queries = {
            "BROKER": ["List all brokers in Kathmandu", "Who are the top 5 brokers?"],
            "FUNDAMENTAL": ["What is the EPS of NABIL bank?", "Show me the dividend history of NRIC"],
            "COMPANY": ["What was the closing price of NABIL on January 15, 2023?", "Tell me about Nabil Bank"],
            "GENERAL": ["What are the steps to apply for an IPO?", "How does NEPSE index calculation work?"],
            "NEPALI": ["काठमाडौंमा कति वटा ब्रोकर छन्?", "नेप्सेमा लगानी गर्न के गर्नुपर्छ?"]
        }
        
        print("\n==== Testing All Benchmark Queries ====")
        for category, queries in test_queries.items():
            print(f"\nTesting {category} category:")
            for i, query in enumerate(queries, 1):
                print(f"\nQuery {i}: {query}")
                self.test_similarity_search(query)
        
        print("\n======================================")
    
    def run_all_diagnostics(self):
        """Run all diagnostic tests"""
        self.test_cuda_extension()
        if self.load_embeddings():
            if self.inspect_vector_stores():
                self.test_similarity_search()
                self.test_all_queries()
        
        print("\n==== Diagnostic Summary ====")
        print("1. Check if any CUDA errors were reported")
        print("2. Verify that vector stores were found and loaded")
        print("3. Examine if similarity searches succeeded")
        print("4. Look for any exception traces in the output")
        print("===========================")
    
    def log_memory_usage(self):
        """Log current memory usage"""
        process = psutil.Process(os.getpid())
        print(f"CPU Memory: {process.memory_info().rss / 1024**2:.2f} MB")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i} Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved / {total:.2f} GB total")


if __name__ == "__main__":
    # Check if paths are provided as arguments
    if len(sys.argv) < 2:
        print("Usage: python test_fixed.py <vector_store_path1> [<vector_store_path2> ...]")
        sys.exit(1)
    
    # Get all vector store paths from command line arguments
    vector_store_paths = sys.argv[1:]
    
    diagnostics = NepseGPTDiagnostics(*vector_store_paths)
    diagnostics.run_all_diagnostics()