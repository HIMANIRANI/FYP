from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from FlagEmbedding import FlagModel
import requests, re, urllib.parse, torch
from threading import Thread
from transformers import BitsAndBytesConfig
import os

import re

def de_camel_case(text: str) -> str:
    # First, handle continuous text without spaces by adding spaces between words
    # This uses regex to find word boundaries between lowercase and uppercase letters
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Add spaces after punctuation
    text = re.sub(r'([.,!?])(?=[A-Za-z])', r'\1 ', text)
    
    # Add spaces between numbers and letters
    text = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])([0-9])', r'\1 \2', text)
    
    # Add spaces between words and numbers in camelCase
    text = re.sub(r'([a-z])([0-9])', r'\1 \2', text)
    
    # Add spaces between words that are run together
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)  # e.g., "wordWord" -> "word Word"
    text = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', text)  # e.g., "WORDWord" -> "WORD Word"
    
    # Ensure proper spacing around punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove spaces before punctuation
    text = re.sub(r'([.,!?])\s+', r'\1 ', text)  # Ensure single space after punctuation
    
    # Handle special case for NEPSE
    text = re.sub(r'NEPSE([a-z])', r'NEPSE \1', text, flags=re.IGNORECASE)
    
    return text

BROKER_TEMPLATE = """
You are NEPSE-GPT, an expert on Nepalese stock brokers. Use ONLY the information provided in the CONTEXT to answer.
Do not invent or infer any detail beyond what's given.

CONTEXT:
{context}

QUESTION:
{question}

BROKER DATABASE FORMAT:
Each entry in CONTEXT has these fields (in order):
  • Broker No
  • Company Name
  • Address (City, District)
  • Website URL
  • TMS URL

INSTRUCTIONS:
1. If the question asks about location, list only brokers whose Address matches exactly.
2. If the question asks for details, present them in bullets:
   • Broker No: [number]
     – Name: [company name]
     – Address: [city, district]
     – Website: [URL]
     – TMS: [URL]
3. If counting is requested, start with "Total: [N] brokers" then list them.
4. If a field is missing, say "Not available" for that field.
5. Keep the answer concise (max 5–7 lines).

ANSWER:
"""

FUNDAMENTAL_TEMPLATE = """
You are NEPSE-GPT, a financial analyst specialized in Nepalese stock fundamentals. Analyze the question using ONLY the provided context which contains the LATEST fundamental data.

CONTEXT:
{context}

QUESTION:
{question}

RESPONSE PROTOCOL:
1. Lead with the most recent metrics first, format: "[Metric]: [Value] [Fiscal Year/Qtr]".
2. Highlight ▲/▼ when comparing to previous data.
3. If data is missing for a requested period, state that explicitly.

TEMPLATE RESPONSE:
"[Company] Fundamentals (Latest):
- EPS: Rs. [X] (FY…)
- P/E: [X] vs sector average
- PBV: [X]
- Dividend: [X]%
- Market Cap: Rs. [X] Cr

Analysis: [Concise interpretation]."

ANSWER:
"""

COMPANY_TEMPLATE = """
You are NEPSE-DataAnalyst, providing strictly fact-based analysis of Nepalese stock market data from 2008-2025.

CONTEXT:
{context}

QUESTION:
{question}

RESPONSE PROTOCOL:
1. For a given date:
   [Company] on [Date]:
     - Closing: Rs. [close] (Range: [min]-[max])
     - Change: [diff] Rs. ([% change]%)
     - Volume: [tradedShares] shares (Rs. [amount])
   • SMA: [SMA]  • RSI: [RSI]  • Bollinger Mid: [BB_Mid]
2. If date missing, explain possible holiday or suspension and show nearest available data.

ANSWER:
"""

PDF_TEMPLATE = """
You are NEPSE-GPT, an expert on Nepalese market rules and procedures. Answer using ONLY the information in CONTEXT.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Keep answers to 3–5 sentences.
- Use bullet points for lists or step-by-step.
- If detail isn't present, say "Based on the available information, I cannot provide details about [topic]."

ANSWER:
"""

class PredictionPipeline:
    def __init__(self):
        # Get the directory where the script is located
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the backend directory (parent of script_dir)
        self.backend_dir = os.path.dirname(self.script_dir)
        # Construct the path to vectordata
        self.vectordata_dir = os.path.join(self.backend_dir, "data", "vectordata")
        
        # At the beginning of your script, before loading models
        try:
            # Force CUDA initialization
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                # Create a small tensor to initialize CUDA
                dummy = torch.ones(1).to(device)
                print(f"CUDA initialized successfully on {torch.cuda.get_device_name(0)}")
            else:
                print("CUDA not available")
        except Exception as e:
            print(f"CUDA initialization error: {str(e)}")
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v0.3" 
        self.temperature = 0.3
        # self.bit = ["gptq-4bit-32g-actorder_True", "gptq-8bit-128g-actorder_True"]
        self.sentence_transformer_modelname = 'sentence-transformers/all-mpnet-base-v2' # 'sentence-transformers/all-MiniLM-L6-v2'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"1. Device being utilized: {self.device} !!!")

    def load_model_and_tokenizers(self):
        ''' Load the TinyLlama model and tokenizer '''
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True,
            model_max_length=2048  # TinyLlama usually uses 2048 max tokens
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        print(f'2. {self.model_id} has been successfully loaded !!!')

    def load_sentence_transformer(self):
        '''
        This method will initialize our sentence transformer model to generate embeddings for a given query.
        '''
        self.sentence_transformer = HuggingFaceEmbeddings(
                                model_name=self.sentence_transformer_modelname,
                                model_kwargs={'device':self.device},
                            )
        print("3. Sentence Transformer Loaded !!!!!!")


    def load_reranking_model(self):
        '''
        An opensoure reranking model called bge-reranker from huggingface is utilized to perform reranking on the retrived relevant documents from vector store.
        This method will initialize the reranking model.        
        '''
        self.reranker = FlagModel('BAAI/bge-reranker-base', use_fp16=True)  # 'BAAI/bge-reranker-large'->2GB BAAI/bge-reranker-base-> 1GB
        print("4. Re-Ranking Algorithm Loaded !!!")
        
    def load_embeddings(self):
        '''
        Load all vector databases with safe deserialization. If databases don't exist, initialize them.
        '''
        try:
            # Broker-related information
            try:
                broker_db_path = os.path.join(self.vectordata_dir, "broker_vector_db")
                self.broker_vector_db = FAISS.load_local(
                    broker_db_path,
                    self.sentence_transformer,
                    allow_dangerous_deserialization=True
                )
                print("Loaded broker vector database")
            except Exception as e:
                print(f"Could not load broker vector database: {e}")
                self.broker_vector_db = None
            
            # Fundamental data of stocks
            try:
                fundamental_db_path = os.path.join(self.vectordata_dir, "fundamental_vector_db")
                self.fundamental_vector_db = FAISS.load_local(
                    fundamental_db_path,
                    self.sentence_transformer,
                    allow_dangerous_deserialization=True
                )
                print("Loaded fundamental vector database")
            except Exception as e:
                print(f"Could not load fundamental vector database: {e}")
                self.fundamental_vector_db = None
            
            # General NEPSE information, rules, IPOs, etc.
            try:
                pdf_db_path = os.path.join(self.vectordata_dir, "pdf_vector_db")
                self.pdf_vector_db = FAISS.load_local(
                    pdf_db_path,
                    self.sentence_transformer,
                    allow_dangerous_deserialization=True
                )
                print("Loaded PDF vector database")
            except Exception as e:
                print(f"Could not load fundamental vector database: {e}")
                self.pdf_vector_db = None
            try:
                company_db_path = os.path.join(self.vectordata_dir, "company_vector_db")
                self.company_vector_db = FAISS.load_local(
                    company_db_path,
                    self.sentence_transformer,
                    allow_dangerous_deserialization=True
                )
                print("Loaded Company vector database")                
            except Exception as e:
                print(f"Could not load Company vector database: {e}")
                self.company_vector_db = None

            # Create a dictionary for easier debugging
            self.faiss_stores = {
                "broker_vector_db": self.broker_vector_db,
                "fundamental_vector_db": self.fundamental_vector_db,
                "pdf_vector_db": self.pdf_vector_db,
                "company_vector_db": self.company_vector_db
            }
            
            # Check which databases were loaded successfully
            loaded_dbs = [name for name, db in self.faiss_stores.items() if db is not None]
            if loaded_dbs:
                print(f"\nSuccessfully loaded vector stores: {', '.join(loaded_dbs)}")
            else:
                print("\nNo vector stores were loaded successfully. You need to initialize them with data first.")
                
            print("\nVector store initialization complete.")
            
        except Exception as e:
            print(f"Error in load_embeddings: {str(e)}")
            raise

    def determine_relevant_db(self, question: str):
        """
        Determine which vector database to query based on the question content.
        Returns the best-matching vector DB based on keyword analysis.
        
        Database categories:
        - broker_vector_db: Broker details (not rules)
        - fundamental_vector_db: Stock fundamental data
        - company_vector_db: Stock details and historical data (2008-2025)
        - pdf_vector_db: General queries, rules, technical information about NEPSE, IPOs
        """
        question_lower = question.lower()
        
        # Define keywords for each category
        category_keywords = {
            "broker": {
                "keywords": ['broker', 'brokerage', 'stock broker', 'broker firm', 'broker detail', 
                            'broker information', 'broker contact', 'broker location', 'broker address',
                            'broker profile', 'list of brokers'],
                "db": self.broker_vector_db
            },
            "fundamental": {
                "keywords": ['fundamental', 'pe ratio', 'eps', 'dividend', 'financial ratio',
                            'book value', 'earning', 'profit', 'revenue', 'balance sheet',
                            'income statement', 'cash flow', 'financial statement', 'market cap',
                            'roa', 'roe', 'debt to equity', 'dividend yield', 'peg ratio'],
                "db": self.fundamental_vector_db
            },
            "company": {
                "keywords": ['company', 'stock', 'share', 'price history', 'trading history', 'stock price',
                            'stock trend', 'stock movement', 'historical data', 'stock performance',
                            'company performance', 'market performance', 'stock chart'] +
                            [str(year) for year in range(2008, 2026)],  # Years 2008-2025
                "db": self.company_vector_db
            },
            "general": {
                "keywords": ['rule', 'regulation', 'ipo', 'nepse', 'made in nepal', 'policy', 'guideline',
                            'requirement', 'procedure', 'trading rule', 'market rule', 'technical',
                            'how to', 'what is', 'process', 'secondary market', 'primary market'],
                "db": self.pdf_vector_db
            }
        }
        
        # Count matches for each category
        category_matches = {category: 0 for category in category_keywords}
        
        for category, info in category_keywords.items():
            for keyword in info["keywords"]:
                if keyword in question_lower:
                    category_matches[category] += 1
        
        # Find category with most keyword matches
        best_match = max(category_matches.items(), key=lambda x: x[1])
        
        # If we have matches, return the corresponding database
        if best_match[1] > 0:
            return category_keywords[best_match[0]]["db"]
        
        # Fallback to general database if no keywords match
        return self.pdf_vector_db

    def rerank_contexts(self, query, contexts, number_of_reranked_documents_to_select=3):
        '''
        Perform reranking on the retrieved documents with special handling for counting queries
        across all entity types (brokers, companies, stocks, etc.)
        '''
        query_lower = query.lower()
        
        # Check if this is a counting query
        counting_patterns = [
            "how many", "count", "total number", "number of", 
            "list all", "show all", "all the", "count of"
        ]
        is_counting_query = any(pattern in query_lower for pattern in counting_patterns)
        
        # Entity types we might want to count
        countable_entities = [
            "broker", "company", "stock", "share", "ipo", 
            "firm", "organization", "corporation", "business",
            "dividend", "sector", "industry", "investor"
        ]
        
        # Check if we're counting any of these entities
        target_entity = None
        for entity in countable_entities:
            if entity in query_lower or f"{entity}s" in query_lower:  # Handle plurals
                target_entity = entity
                break
        
        # Potential locations or filters in the query
        filters = [
            "kathmandu", "lalitpur", "bhaktapur", "pokhara", 
            "nepal", "valley", "city", "location", "area", "region",
            "sector", "industry", "profitable", "dividend", "year",
            "2023", "2024", "2025", "active", "licensed"
        ]
        
        has_filter = any(filter_term in query_lower for filter_term in filters)
        
        # For counting queries with entities, retrieve more documents
        if is_counting_query and target_entity:
            # Increase the number based on whether there's a filter
            if has_filter:
                # With filters, we need more docs to ensure accurate counting
                retrieval_count = min(100, len(contexts))
            else:
                # For broad counts, we may need almost all documents
                retrieval_count = min(200, len(contexts))
                
            # For very specific questions that might need all data
            if "all" in query_lower or "every" in query_lower:
                retrieval_count = len(contexts)  # Get all available contexts
        else:
            # For normal queries, use the specified number
            retrieval_count = number_of_reranked_documents_to_select
        
        # Standard reranking process
        embeddings_1 = self.reranker.encode(query)
        embeddings_2 = self.reranker.encode(contexts)
        similarity = embeddings_1 @ embeddings_2.T

        number_of_contexts = len(contexts)
        if retrieval_count > number_of_contexts:
            retrieval_count = number_of_contexts

        highest_ranked_indices = sorted(range(len(similarity)), 
                                    key=lambda i: similarity[i], 
                                    reverse=True)[:retrieval_count]
        return [contexts[index] for index in highest_ranked_indices]
    

    def is_text_nepali(self, text):
        '''
        This method checks if a question asked by the user contains any nepali word. If so, the response from the LLM is also returned in Nepali -
        - using google translate API

        parameters:
        text -> the question asked by the user

        returns: bool
        True if the text contains any nepali word else false
        '''
        nepali_regex = re.compile(r'[\u0900-\u097F]+')
        return bool(nepali_regex.search(text))
    

    def translate_using_google_api(self, text, source_language = "auto", target_language = "ne", timeout=5):
        '''
        This function has been copied from here:
        # https://github.com/ahmeterenodaci/easygoogletranslate/blob/main/easygoogletranslate.py

        This free API is used to perform translation between English to Nepali and vice versa.

        parameters: 
        source_language -> the language code for the source language
        target_language -> the new language to which the text is to be translate 

        returns
        '''
        pattern = r'(?s)class="(?:t0|result-container)">(.*?)<'
        escaped_text = urllib.parse.quote(text.encode('utf8'))
        url = 'https://translate.google.com/m?tl=%s&sl=%s&q=%s'%(target_language, source_language, escaped_text)
        response = requests.get(url, timeout=timeout)
        result = response.text.encode('utf8').decode('utf8')
        result = re.findall(pattern, result)  
        return result
    
    def split_and_translate_text(self, text, source_language = "auto", target_language = "ne", max_length=5000):
        """
        Split the input text into sections with a maximum length.
        
        Parameters:
        - text: The input text to be split.
        - max_length: The maximum length for each section (default is 5000 characters).

        Returns:c
        A list of strings, each representing a section of the input text.
        """

        if source_language == "en":
            splitted_text = text.split(".")
        elif source_language == "ne":
            splitted_text = text.split("।")
        else:
            splitted_text = [text[i:i+max_length] for i in range(0, len(text), max_length)]

        # perform translation (the free google api can only perform translation for 5000 characters max. So, splitting the text is necessary )
        translate_and_join_splitted_text = " ".join([self.translate_using_google_api(i, source_language, target_language)[0] for i in splitted_text])
        return translate_and_join_splitted_text
    
    def perform_translation(self, question, source_language, target_language):
        try:
            # Check if the length of the question is greater than 5000 characters
            if len(question) > 5000:
                # If so, split and translate the text using a custom method
                return self.split_and_translate_text(question, source_language, target_language)
            else:
                # If not, use the Google Translation API to translate the entire text
                return self.translate_using_google_api(question, source_language, target_language)[0]
        except Exception as e:
            return [f"An error occurred, [{e}], while working with Google Translation API"]
        
    def make_predictions(self, question, top_n_values=10):
        '''
        Main prediction method with enhanced database-specific prompts for NEPSE data

        Args:
            question: User's question text (can be in English or Nepali)
            top_n_values: Maximum number of documents to retrieve initially

        Returns:
            Generator yielding response tokens for streaming
        '''
        question = de_camel_case(question)  # Only apply to question here

        try:
            # Check if question is in Nepali
            is_original_language_nepali = self.is_text_nepali(question)

            # Translate if Nepali
            if is_original_language_nepali:
                try:
                    question = self.perform_translation(question, 'ne', 'en')
                    print("Translated Question: ", question)
                    if isinstance(question, list):
                        yield "data: " + str(question[0]) + "\n\n"
                        yield "data: END\n\n"
                        return
                except Exception as e:
                    print(f"Translation error: {e}")
                    yield f"data: Sorry, I encountered an error translating your question. Please try asking in English.\n\n"
                    yield "data: END\n\n"
                    return

            # Determine which vector DB to use
            try:
                vector_db = self.determine_relevant_db(question)
            except Exception as e:
                print(f"Database selection error: {e}")
                vector_db = self.pdf_vector_db  # Fallback to general DB

            # Get relevant documents
            try:
                similarity_search = vector_db.similarity_search_with_score(question, k=top_n_values)
                context = [doc.page_content for doc, score in similarity_search if score < 1.5]
                number_of_contexts = len(context)

                if number_of_contexts == 0:
                    yield "data: I couldn't find relevant information to answer your question. Please try rephrasing or ask about NEPSE, stocks, brokers, or related topics.\n\n"
                    yield "data: END\n\n"
                    return

                if number_of_contexts > 1:
                    context = self.rerank_contexts(question, context)

                context = ". ".join(context)
                context = de_camel_case(context)  # ✅ Now it's safe to use
            except Exception as e:
                print(f"Context retrieval error: {e}")
                yield "data: I encountered an error retrieving the necessary information. Please try again later.\n\n"
                yield "data: END\n\n"
                return

            # Determine database type and create appropriate prompt
            db_type = "general"
            if vector_db == self.broker_vector_db:
                db_type = "broker"
            elif vector_db == self.fundamental_vector_db:
                db_type = "fundamental"
            elif vector_db == self.company_vector_db:
                db_type = "company"

            # Check for counting queries
            query_lower = question.lower()
            counting_patterns = ["how many", "count", "total number", "number of", "list all", "show all", "all the", "count of"]
            is_counting_query = any(pattern in query_lower for pattern in counting_patterns)

            if is_counting_query:
                counting_prompt = '''
                You are NEPSE-GPT, tasked with counting or listing items from the Nepalese stock market. Answer the question using ONLY the information provided in the context below.

                CONTEXT INFORMATION:
                {context}

                QUESTION: {question}

                INSTRUCTIONS:
                1. Your primary task is to COUNT or LIST items accurately based on the context.
                2. Explicitly state the total count at the beginning of your answer (e.g., "There are 12 brokers in Kathmandu").
                3. If the context contains a partial list, clearly state this limitation (e.g., "Based on the available information, I can identify at least 8 companies...").
                4. For filtered counting queries (e.g., brokers in a specific location), specify both the filter criteria and count.
                5. If appropriate, briefly list the items being counted (especially for small counts).
                6. If you cannot determine an exact count from the context, provide the best estimate based solely on the information provided and explain your uncertainty.
                7. Do not make up or infer information not present in the context.
                8. Keep your answer concise but ensure counting accuracy is the top priority.

                ANSWER:
                '''
                prompt = counting_prompt.format(question=question, context=context)
            else:
                if db_type == "broker":
                    prompt = BROKER_TEMPLATE.format(context=context, question=question)
                elif db_type == "fundamental":
                    prompt = FUNDAMENTAL_TEMPLATE.format(context=context, question=question)
                elif db_type == "company":
                    prompt = COMPANY_TEMPLATE.format(context=context, question=question)
                else:
                    prompt = PDF_TEMPLATE.format(context=context, question=question)

            print("----- Prompt Preview -----")
            print(prompt[:1000])
            print("--------------------------")

            try:
                inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

                generation_kwargs = dict(
                    inputs,
                    streamer=self.streamer,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.95,
                    top_k=40,
                    repetition_penalty=1.1,
                    pad_token_id=50256
                )

                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                if is_original_language_nepali:
                    sentence = ""
                    for token in self.streamer:
                        if token != "</s>":
                            sentence += token
                            if "." in token:
                                try:
                                    translated = self.translate_using_google_api(sentence, "en", "ne")[0]
                                    translated = re.sub(r'</?s>', '', translated)
                                    yield f"data: {translated}\n\n"
                                    sentence = ""
                                except Exception as e:
                                    print(f"Translation error during streaming: {e}")
                                    yield f"data: {sentence}\n\n"
                                    sentence = ""
                else:
                    for token in self.streamer:
                        yield f"data: {token}\n\n"

                thread.join()
                yield "data: END\n\n"

            except Exception as e:
                print(f"Response generation error: {e}")
                yield "data: I encountered a technical issue while generating a response. Please try again later.\n\n"
                yield "data: END\n\n"
                return

        except Exception as e:
            print(f"Unexpected error in make_predictions: {e}")
            yield "data: Sorry, an unexpected error occurred. Please try again later.\n\n"
            yield "data: END\n\n"
            return

        
        
# if __name__ == "__main__":
#     # Instantiate the pipeline
#     pipeline = PredictionPipeline()
    
#     # Load everything
#     pipeline.load_model_and_tokenizers()
#     pipeline.load_sentence_transformer()
#     pipeline.load_reranking_model()
#     pipeline.load_embeddings()

#     # ✅ Multiple test queries mapped to intended vector stores
#     test_queries = {
#         "broker_vector_db": "Which brokers are located in Kathmandu?",
#         "fundamental_vector_db": "What is the EPS of NABIL bank?",
#         "company_vector_db": "Show me the stock price history of NLIC in 2020.",
#         "pdf_vector_db": "What are the trading rules in NEPSE?"
#     }

#     # ✅ Loop through and test each
#     for db_name, test_question in test_queries.items():
#         print(f"\n=== Testing Query for {db_name} ===")
#         print(f"Test Question: {test_question}\n")
        
#         response = ""
#         for output in pipeline.make_predictions(test_question):
#             if output.startswith("data: "):
#                 content = output.replace("data: ", "").strip()
#                 if content == "END":
#                     break
#                 response += content + " "
        
#         print("Final Answer:\n")
#         print(response.strip())
#         print("\n" + "="*80 + "\n")