import re
import urllib.parse
from pathlib import Path
from threading import Thread
from typing import Generator

import requests
import torch
from FlagEmbedding import FlagModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langdetect import DetectorFactory, detect
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)

# Ensure consistent language detection
DetectorFactory.seed = 0

class EnhancedPredictionPipeline:
    def __init__(self):
        # Model configuration
        self.model_id = "TheBloke/neural-chat-7B-v3-1-GPTQ"
        self.temperature = 0.3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # VectorDB paths
        base_dir = Path(__file__).resolve().parent.parent
        vector_data_dir = base_dir / "data" / "vector data"
        self.vector_db_paths = {
            "broker": vector_data_dir / "broker_vector_db",
            "fundamental": vector_data_dir / "fundamental_vector_db",
            "pdf": vector_data_dir / "pdf_vector_db",
            "company": vector_data_dir / "company_vector_db"
        }
        
        # Initialize components
        self.load_components()
        self.load_vector_dbs()
        
    def load_components(self):
        """Initialize models and embeddings"""
        # Tokenizer and LLM
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            device_map=self.device,
            model_max_length=4000,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device,
            trust_remote_code=True
        )
        
        # Embeddings and reranker
        self.embedder = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-mpnet-base-v2',
            model_kwargs={'device': self.device}
        )
        self.reranker = FlagModel('BAAI/bge-reranker-large', use_fp16=True)
        self.streamer = TextIteratorStreamer(self.tokenizer)

    def load_vector_dbs(self):
        """Load all vector databases"""
        self.vector_dbs = {}
        for db_name, path in self.vector_db_paths.items():
            try:
                self.vector_dbs[db_name] = FAISS.load_local(
                    str(path),
                    self.embedder,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading {db_name} DB: {e}")

    def detect_language(self, text: str) -> str:
        """Robust language detection with fallback"""
        try:
            return detect(text)
        except:
            return 'en'  # Default to English

    def is_text_nepali(self, text: str) -> bool:
        """Check for Nepali characters"""
        nepali_regex = re.compile(r'[\u0900-\u097F]+')
        return bool(nepali_regex.search(text))

    def translate_using_google_api(self, text: str, src: str, tgt: str) -> str:
        """Direct translation using Google's web interface"""
        pattern = r'(?s)class="(?:t0|result-container)">(.*?)<'
        escaped_text = urllib.parse.quote(text.encode('utf8'))
        url = f'https://translate.google.com/m?tl={tgt}&sl={src}&q={escaped_text}'
        try:
            response = requests.get(url, timeout=5)
            result = response.text.encode('utf8').decode('utf8')
            matches = re.findall(pattern, result)
            return matches[0] if matches else text
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def perform_translation(self, text: str, src: str, tgt: str) -> str:
        """Batch translation with chunking"""
        if len(text) <= 5000:
            return self.translate_using_google_api(text, src, tgt)
        
        # Split and translate chunks
        chunks = []
        if src == "en":
            split_points = [m.end() for m in re.finditer(r'\.\s+', text)]
        elif src == "ne":
            split_points = [m.end() for m in re.finditer(r'।\s+', text)]
        else:
            split_points = list(range(0, len(text), 4500))
            
        prev = 0
        for end in split_points + [len(text)]:
            chunk = text[prev:end]
            if chunk.strip():
                chunks.append(self.translate_using_google_api(chunk, src, tgt))
            prev = end
            
        return ' '.join(chunks)

    def select_relevant_contexts(contexts, max_chars=1500):
        selected = []
        current_length = 0
        
        for ctx in contexts:
            if current_length + len(ctx) > max_chars:
                break
            selected.append(ctx)
            current_length += len(ctx)
        
        return selected

    FINANCIAL_GLOSSARY = {
        "P/E ratio": "P/E अनुपात",
        "EPS": "प्रति शेयर आम्दानी",
        "Market Cap": "बजार पूँजी"
    }

    def translate_financial_terms(text):
        for eng, nep in FINANCIAL_GLOSSARY.items():
            text = text.replace(eng, nep)
        return text

    def unified_retrieval(self, query: str, top_k: int = 10) -> list:
        """Search across all vector databases"""
        all_results = []
        for db_name, vector_db in self.vector_dbs.items():
            try:
                docs = vector_db.similarity_search_with_score(query, k=top_k)
                all_results.extend([
                    (doc.page_content, score, db_name)
                    for doc, score in docs
                    if score < 1.5  # Similarity threshold
                ])
            except Exception as e:
                print(f"Search failed in {db_name}: {e}")
        
        # Rerank globally
        if all_results:
            contents = [res[0] for res in all_results]
            # Sort results by score in ascending order (lower is better)
            sorted_results = sorted(
                enumerate(all_results),
                key=lambda x: x[1][1]
            )
            top_results = sorted_results[:top_k]
            return [
                (all_results[i][0], all_results[i][1], all_results[i][2])
                for i, _ in top_results
            ]
        return []

    def build_contextual_prompt(self, query: str, contexts: list) -> str:
        """Construct domain-aware prompt"""
        context_str = "\n\n".join([
            f"[Source: {source}]\n{content}"
            for content, _, source in contexts
        ])
        
        return f"""As a NEPSE financial expert, answer using ONLY these sources:
        
{context_str}

Question: {query}
Answer concisely in 3-5 sentences. If unsure, state you don't know.
Answer:"""

    def generate_response(self, query: str) -> Generator[str, None, None]:
        """Core response generation pipeline"""
        # Phase 1: Context retrieval
        contexts = self.unified_retrieval(query)
        if not contexts:
            yield "data: I couldn't find relevant information to answer this question.\n\n"
            yield "data: END\n\n"
            return
        
        # Phase 2: Prompt construction
        prompt = self.build_contextual_prompt(query, contexts)
        
        # Phase 3: Stream generation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generation_kwargs = dict(
            inputs,
            streamer=self.streamer,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2
        )
        
        # Start generation thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream tokens
        buffer = ""
        for token in self.streamer:
            buffer += token
            if any(punct in buffer for punct in [".", "!", "?", "\n", ":", ","]):
                yield f"data: {buffer.strip()}\n\n"
                buffer = ""
        
        # Flush remaining buffer
        if buffer:
            yield f"data: {buffer.strip()}\n\n"
        
        yield "data: END\n\n"
        thread.join()

class BilingualPipeline(EnhancedPredictionPipeline):
    def __init__(self):
        super().__init__()
        self.translation_cache = {}  # Simple translation cache

    def generate_response(self, query: str) -> Generator[str, None, None]:
        # Detect input language
        src_lang = self.detect_language(query)
        
        # Translate query if needed
        if src_lang == 'ne':
            if query in self.translation_cache:
                translated_query = self.translation_cache[query]
            else:
                translated_query = self.perform_translation(query, 'ne', 'en')
                self.translation_cache[query] = translated_query
        else:
            translated_query = query

        # Get base English response stream
        english_stream = super().generate_response(translated_query)
        
        # Process stream based on language
        buffer = ""
        for chunk in english_stream:
            if "END" in chunk:
                if src_lang == 'ne' and buffer:
                    # Translate final buffer
                    try:
                        translated = self.perform_translation(buffer, 'en', 'ne')
                        yield f"data: {translated}\n\n"
                    except Exception as e:
                        yield f"data: (Translation error: {str(e)})\n\n"
                yield "data: END\n\n"
                break
            
            # Accumulate tokens
            buffer += chunk.replace("data: ", "").strip()
            
            # Check for natural translation points
            if any(punct in buffer for punct in [".", "!", "?", "\n", "।"]):
                if src_lang == 'ne':
                    try:
                        translated = self.perform_translation(buffer, 'en', 'ne')
                        yield f"data: {translated}\n\n"
                    except Exception as e:
                        yield f"data: (Translation error: {str(e)})\n\n"
                    buffer = ""
                else:
                    yield f"data: {buffer}\n\n"
                    buffer = ""

# Usage Example
if __name__ == "__main__":
    pipeline = BilingualPipeline()
    
    # Test Nepali query
    nepali_query = "नेप्से भनेको के हो ?"
    print(f"Query: {nepali_query}")
    for response in pipeline.generate_response(nepali_query):
        print(response)
    
    # Test English query
    english_query = "Show me NWCL's recent stock price trends"
    print(f"\nQuery: {english_query}")
    for response in pipeline.generate_response(english_query):
        print(response)
