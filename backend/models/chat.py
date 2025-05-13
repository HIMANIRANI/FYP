from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig
)
from transformers import pipeline as hf_pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from huggingface_hub import login
import torch
import os
import re
import logging
import time
import psutil
import requests
import json
import random
from threading import Thread
from typing import List, Dict
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Authenticate with Hugging Face
if os.getenv("HF_TOKEN"):
    logger.info("Authenticating with Hugging Face")
    login(token=os.getenv("HF_TOKEN"))
    print(f"Token loaded: {os.getenv('HF_TOKEN') is not None}")
else:
    logger.warning("HF_TOKEN not found in environment variables")

# Translation is now available with the free implementation
print("✅ Translation features are enabled using free Google Translate")

def get_system_memory_info():
    """Get system memory information"""
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024 ** 3)
    available_gb = memory.available / (1024 ** 3)
    used_gb = memory.used / (1024 ** 3)
    
    print(f"System Memory: Total={total_gb:.2f}GB, Used={used_gb:.2f}GB, Available={available_gb:.2f}GB")
    return memory.total, memory.available

class TranslationService:
    """Free Google Translate service using web scraping"""
    def __init__(self):
        self.base_url = "https://translate.google.com/m"
        self.available = True
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
        ]
        
    def translate(self, text: str, source_lang: str = "ne", target_lang: str = "en") -> str:
        """Translate text using free Google Translate web interface"""
        if not text.strip():
            return text
            
        try:
            # Add a small delay to avoid rate limiting
            time.sleep(random.uniform(0.1, 0.5))
            
            # Encode text for URL
            encoded_text = quote(text)
            
            # Construct URL
            url = f"{self.base_url}?sl={source_lang}&tl={target_lang}&q={encoded_text}"
            
            # Random user agent to avoid blocking
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://translate.google.com/',
                'DNT': '1',
            }
            
            # Make request
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                # Extract translation using regex
                # Pattern looks for content within the translation result div
                result = re.search(r'class="result-container">(.*?)</div>', response.text)
                
                if result:
                    translated_text = result.group(1)
                    # Clean up HTML entities
                    translated_text = translated_text.replace("&amp;", "&")
                    translated_text = translated_text.replace("&lt;", "<")
                    translated_text = translated_text.replace("&gt;", ">")
                    translated_text = translated_text.replace("&quot;", "\"")
                    translated_text = translated_text.replace("&#39;", "'")
                    return translated_text
                else:
                    logger.warning("Translation pattern not found in response")
                    return text
            else:
                logger.error(f"Translation request failed: {response.status_code}")
                return text
                
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text
            
    def detect_language(self, text: str) -> str:
        """Basic language detection - just checks for Devanagari script for Nepali"""
        nepali_range = re.compile(r'[\u0900-\u097F]')
        return "ne" if bool(nepali_range.search(text)) else "en"

class PredictionPipeline:
    def __init__(self, use_cpu=False, use_8bit=False, use_4bit=False, offload_folder="./offload"):
        # Device and memory configuration
        self.use_cpu = use_cpu
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        self.offload_folder = offload_folder
        os.makedirs(offload_folder, exist_ok=True)
        
        # Translation service
        self.translator = TranslationService()
        
        # Model configurations - Using smaller models
        self.llm_model_id = "google/gemma-2b-it"  # Consider "pfnet/plamo-1.1b" for even smaller model
        self.classifier_model_id = "distilbert-base-uncased"  
        self.sentence_transformer_modelname = 'sentence-transformers/all-MiniLM-L6-v2'
        self.reranker_model_id = "cross-encoder/ms-marco-TinyBERT-L-2"
        
        # Model placement strategy
        self.model_placement = {
            'llm': 'gpu',  # Always try to put LLM on GPU
            'classifier': 'cpu',
            'reranker': 'cpu',
            'embedder': 'cpu'
        }
        
        # Setup device
        if torch.cuda.is_available() and not use_cpu:
            self.device = torch.device('cuda')
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)} with {gpu_mem:.2f}GB memory")
            
            # If GPU memory is very limited, adjust placement
            if gpu_mem < 6:  # Less than 6GB
                self.model_placement['llm'] = 'gpu'  # Still try for LLM
        else:
            self.device = torch.device('cpu')
            # Force all models to CPU if no GPU or use_cpu=True
            self.model_placement = {k: 'cpu' for k in self.model_placement.keys()}
            logger.info("Using CPU for computation")
        
        # Vector store paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.vector_db_paths = {
            'broker': os.path.join(base_dir, "data/vector data/broker_vector_db"),
            'fundamental': os.path.join(base_dir, "data/vector data/fundamental_vector_db"),
            'company': os.path.join(base_dir, "data/vector data/company_vector_db"),
            'pdf': os.path.join(base_dir, "data/vector data/pdf_vector_db")
        }
        
        # Initialize model containers
        self.llm_model = None
        self.llm_tokenizer = None
        self.classifier = None
        self.reranker = None
        self.embedder = None
        self.vector_dbs = {}
        
        # Simplified prompt templates
        self.prompt_templates = {
            "broker_contact": """
            <|system|>
            You are a broker information assistant. Provide details from context only.</s>
            <|user|>
            Context: {context}
            Question: {question}</s>
            <|assistant|>
            """,
            "stock_info": """
            <|system|>
            You are a stock analyst. Provide precise data.</s>
            <|user|>
            Context: {context}
            Question: {question}</s>
            <|assistant|>
            """,
            "regulations": """
            <|system|>
            You are a regulatory expert. explain them in layman terms</s>
            <|user|>
            Context: {context}
            Question: {question}</s>
            <|assistant|>
            """,
            "fundamental": """
            <|system|>
            Explain financial concepts simply.</s>
            <|user|>
            Context: {context}
            Question: {question}</s>
            <|assistant|>
            """
        }

    def _get_quantization_config(self):
        """Get appropriate quantization configuration based on settings"""
        if self.use_cpu:
            return None
        elif self.use_8bit:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
        elif self.use_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        return None

    def load_tokenizers(self):
        """Load tokenizers separately to reduce memory pressure"""
        try:
            logger.info("Loading tokenizers...")
            
            # LLM tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_id)
            
            logger.info("Tokenizers loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading tokenizers: {str(e)}")
            return False

    def load_small_models(self):
        """Load smaller models all on CPU"""
        try:
            # Load embedder (always CPU)
            logger.info("Loading sentence transformer...")
            self.embedder = HuggingFaceEmbeddings(
                model_name=self.sentence_transformer_modelname,
                model_kwargs={'device': 'cpu'}
            )
            
            # Load classifier (always CPU)
            logger.info("Loading DistilBERT classifier...")
            self.classifier = hf_pipeline(
                "text-classification",
                model=self.classifier_model_id,
                device=-1  # Force CPU
            )
            
            # Load reranker (always CPU)
            logger.info("Loading TinyBERT reranker...")
            self.reranker = CrossEncoder(
                self.reranker_model_id, 
                device='cpu'
            )
            
            logger.info("All small models loaded on CPU")
            return True
        except Exception as e:
            logger.error(f"Error loading small models: {str(e)}")
            return False

    def load_llm(self):
        """Load LLM model with priority on GPU"""
        try:
            logger.info("Loading LLM model...")
            
            # Determine device placement
            llm_device = 'cuda' if self.model_placement['llm'] == 'gpu' and torch.cuda.is_available() else 'cpu'
            
            quantization_config = self._get_quantization_config() if llm_device == 'cuda' else None
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_id,
                device_map=llm_device if llm_device == 'cuda' else None,
                torch_dtype=torch.float16 if llm_device == 'cuda' else torch.float32,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True
            )
            
            # Explicitly move to device if not using device_map
            if llm_device != 'cuda':
                self.llm_model = self.llm_model.to(llm_device)
                
            logger.info(f"LLM loaded successfully on {llm_device.upper()}")
            return True
        except Exception as e:
            logger.error(f"Error loading LLM: {str(e)}")
            # Fallback to CPU if GPU fails
            if llm_device == 'cuda':
                logger.warning("Attempting fallback to CPU for LLM")
                self.model_placement['llm'] = 'cpu'
                return self.load_llm()
            return False

    def load_models(self):
        """Load all models with memory optimization"""
        # Clear cache before loading
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        success = True
        
        # Stage 1: Load tokenizers
        if not self.load_tokenizers():
            logger.error("Failed to load tokenizers")
            success = False
        
        # Stage 2: Load small CPU models first
        if success and not self.load_small_models():
            logger.error("Failed to load small models")
            success = False
            
        # Stage 3: Try loading LLM on GPU (with fallback to CPU)
        if success and not self.load_llm():
            logger.error("Failed to load LLM")
            success = False
            
        return success

    def load_vector_dbs(self):
        """Load all FAISS vector stores with error handling"""
        self.vector_dbs = {}
        for name, path in self.vector_db_paths.items():
            try:
                if os.path.exists(path):
                    self.vector_dbs[name] = FAISS.load_local(
                        path,
                        self.embedder,
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"Loaded {name} DB with {len(self.vector_dbs[name].index_to_docstore_id)} docs")
                else:
                    logger.warning(f"Vector DB not found at {path}")
            except Exception as e:
                logger.error(f"Failed loading {name} DB: {e}")

    def translate_text(self, text: str, source_lang: str = "ne", target_lang: str = "en") -> str:
        """Translate text using Google Translate"""
        return self.translator.translate(text, source_lang, target_lang)

    def classify_query(self, query: str) -> str:
        """Classify queries into appropriate categories based on content.
        
        Categories and their corresponding databases:
        - broker_contact: Broker contact details only
        - stock_info: Stock prices and technical indicators (company DB)
        - fundamental: Fundamental analysis data (fundamental DB)
        - regulations: NEPSE, CDSC rules and regulations (pdf DB)
        """
        try:
            # If classifier not loaded, default to fundamental
            if not self.classifier:
                return "fundamental"
                
            # First check for specific patterns that should override classifier
            query_lower = query.lower()
            
            # Check for broker contact queries
            broker_contact_keywords = [
                "contact", "phone", "email", "address", "broker details",
                "how to reach", "where is", "location of","broker"
            ]
            if any(keyword in query_lower for keyword in broker_contact_keywords):
                return "broker_contact"
                
            # Check for NEPSE/CDSC regulations
            regulation_keywords = [
                "nepse", "cdsc", "regulation", "rule", "policy", 
                "procedure", "guideline", "requirement", "compliance",
                "what is nepse", "explain nepse", "nepse definition"
            ]
            if any(keyword in query_lower for keyword in regulation_keywords):
                return "regulations"
                
            # Check for fundamental analysis queries
            fundamental_keywords = [
                "fundamental", "pe ratio", "eps", "dividend", "book value",
                "financial", "balance sheet", "income statement", "annual report"
            ]
            if any(keyword in query_lower for keyword in fundamental_keywords):
                return "fundamental"
                
            # Default to using the classifier for remaining cases
            categories = {
                "LABEL_0": "broker_contact",
                "LABEL_1": "stock_info",
                "LABEL_2": "regulations",
                "LABEL_3": "fundamental"
            }
            result = self.classifier(query[:512])  # Truncate to model max length
            
            # Special handling for stock info vs fundamental
            if result[0]["label"] == "LABEL_1":  # stock_info
                # Check if it's actually a fundamental query
                stock_tech_keywords = [
                    "price", "close", "open", "high", "low", 
                    "rsi", "macd", "sma", "ema", "bollinger",
                    "volume", "technical", "chart", "indicator"
                ]
                if not any(keyword in query_lower for keyword in stock_tech_keywords):
                    return "fundamental"
                    
            return categories.get(result[0]["label"], "fundamental")
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return "fundamental"  # Default to fundamental query type
    
    def retrieve_context(self, query: str, query_type: str, k: int = 3) -> List[str]:
        """Retrieve relevant context - reduced from 5 to 3 for efficiency"""
        try:
            db_map = {
                "broker_contact": "broker",
                "stock_info": "company",
                "regulations": "pdf",
                "fundamental": "fundamental"
            }
            
            if query_type not in db_map:
                logger.warning(f"Unknown query type: {query_type}")
                return []
            
            db = self.vector_dbs.get(db_map[query_type])
            if not db:
                logger.error(f"No DB available for {query_type}")
                return []
            
            # First-stage retrieval
            docs = db.similarity_search(query, k=k*2)
            contexts = [doc.page_content for doc in docs]
            
            # Second-stage reranking only if reranker is loaded
            if len(contexts) > 1 and self.reranker:
                pairs = [(query, ctx) for ctx in contexts]
                scores = self.reranker.predict(pairs)
                contexts = [ctx for _, ctx in sorted(zip(scores, contexts), reverse=True)]
            
            return contexts[:k]
        except Exception as e:
            logger.error(f"Context retrieval error: {str(e)}")
            return []

    def generate_response(self, query: str, query_type: str, context: str) -> str:
        """Generate response using LLM with optimized prompts"""
        try:
            # If LLM not loaded, return fallback
            if not self.llm_model or not self.llm_tokenizer:
                return self._get_fallback_response(query_type)
                
            prompt_template = self.prompt_templates.get(query_type, self.prompt_templates["fundamental"])
            prompt = prompt_template.format(context=context, question=query)
            
            inputs = self.llm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Move to model's device
            if hasattr(self.llm_model, 'device'):
                inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
                
            # Use reasonable number of tokens for responses
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=256,  # Increased for better responses
                temperature=0.3,
                top_p=0.9
            )
            
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
                
            return response
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return self._get_fallback_response(query_type)

    def process_query(self, query: str) -> str:
        """End-to-end query processing with memory efficiency focus"""
        try:
            # Language detection and translation
            original_language = "nepali" if self.translator.detect_language(query) == "ne" else "english"
            
            if original_language == "nepali" and self.translator.available:
                english_query = self.translate_text(query, source_lang="ne", target_lang="en")
                logger.info(f"Translated query to English: {english_query}")
                query = english_query  # Use translated query for processing
            
            # Query classification
            query_type = self.classify_query(query)
            logger.info(f"Classified as: {query_type}")
            
            # Context retrieval
            contexts = self.retrieve_context(query, query_type)
            if not contexts:
                logger.warning(f"No context found for query: {query}")
                response = self._get_fallback_response(query_type)
            else:
                # Join contexts with separator for clarity
                context = ". ".join(contexts)
                logger.info(f"Retrieved {len(contexts)} context chunks")
                
                # Response generation
                response = self.generate_response(query, query_type, context)
            
            # Translate back if original was Nepali and translator is available
            if original_language == "nepali" and self.translator.available:
                nepali_response = self.translate_text(response, source_lang="en", target_lang="ne")
                logger.info("Translated response back to Nepali")
                return nepali_response
            
            return response
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "माफ गर्नुहोस्, म अहिले जवाफ दिन सक्दिन।" if self.translator.detect_language(query) == "ne" else "Sorry, I'm unable to respond right now."

    def _get_fallback_response(self, query_type: str) -> str:
        """Rule-based fallback when no context is found"""
        fallbacks = {
            "broker_contact": "Broker details not found. Visit SEBON's website for updated listings.",
            "stock_info": "Current stock data unavailable. Check NEPSE's official portal for live updates.",
            "regulations": "Regulation not found in my database. Refer to SEBON's latest circulars.",
            "fundamental": "I couldn't find a precise answer. Generally in NEPSE..."
        }
        return fallbacks.get(query_type, "Information not available.")

    def unload_models(self):
        """Unload models to free memory"""
        logger.info("Unloading models to free memory...")
        
        # Unload LLM
        if self.llm_model:
            del self.llm_model
            self.llm_model = None
            
        # Unload other models
        if self.classifier:
            del self.classifier
            self.classifier = None
            
        if self.reranker:
            del self.reranker
            self.reranker = None
            
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Models unloaded successfully")

def main():
    # Check system memory
    total_mem, available_mem = get_system_memory_info()
    
    # Check GPU memory
    gpu_available = torch.cuda.is_available()
    gpu_memory_gb = 0
    
    if gpu_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory / (1024**3)
        print(f"GPU Memory: {gpu_memory_gb:.2f}GB")
    
    # Determine best configuration based on available hardware
    use_cpu = not gpu_available
    use_8bit = False  # Default to not use 8-bit quantization
    use_4bit = True   # Default to 4-bit quantization
    
    if gpu_available:
        if gpu_memory_gb < 4:  # Very limited GPU memory
            print("⚠️ Extremely limited GPU memory detected. Using CPU instead...")
            use_cpu = True
            use_4bit = False
        elif gpu_memory_gb < 6:  # Limited GPU memory
            print("⚠️ Limited GPU memory detected. Using 4-bit quantization...")
            use_4bit = True
        elif gpu_memory_gb < 12:  # Moderate GPU memory
            print("⚠️ Moderate GPU memory detected. Using 8-bit quantization...")
            use_8bit = True
            use_4bit = False
        else:
            print("✅ Sufficient GPU memory detected. Using FP16.")
            use_8bit = False
            use_4bit = False
    else:
        print("⚠️ No GPU detected. Using CPU mode.")
    
    # Create pipeline
    pipeline = PredictionPipeline(use_cpu=use_cpu, use_8bit=use_8bit, use_4bit=use_4bit)
    
    if pipeline.load_models():
        pipeline.load_vector_dbs()
        
        try:
            # Interactive mode
            print("\n=== NEPSE-Navigator Chat ===")
            print("Type 'exit' to quit")
            
            while True:
                query = input("\nQuery: ")
                if query.lower() in ['exit', 'quit', 'q']:
                    break
                    
                print("Processing...")
                response = pipeline.process_query(query)
                print(f"Response: {response}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            # Clean up
            pipeline.unload_models()
    else:
        print("Failed to load models. Please check the logs for more information.")

if __name__ == "__main__":
    main()