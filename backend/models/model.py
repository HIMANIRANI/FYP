from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from FlagEmbedding import FlagModel
import requests, re, urllib.parse, torch
from threading import Thread


class PredictionPipeline:
    def __init__(self):
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
        self.model_id = "TheBloke/neural-chat-7B-v3-1-GPTQ" #'TheBloke/Starling-LM-7B-alpha-GPTQ' 
        self.temperature = 0.3
        self.bit = ["gptq-4bit-32g-actorder_True", "gptq-8bit-128g-actorder_True"]
        self.sentence_transformer_modelname = 'sentence-transformers/all-mpnet-base-v2' # 'sentence-transformers/all-MiniLM-L6-v2'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"1. Device being utilized: {self.device} !!!")


    def load_model_and_tokenizers(self):
        '''
        This method will initialize the tokenizer and our LLM model and the streamer class.
        '''
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, torch_dtype=torch.float16, device_map=self.device,  use_fast=True, model_max_length=4000)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id,  device_map=self.device, trust_remote_code=False,
                                                          revision=self.bit[1]) 
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
        self.reranker = FlagModel('BAAI/bge-reranker-large', use_fp16=True)  # 'BAAI/bge-reranker-large'->2GB BAAI/bge-reranker-base-> 1GB
        print("4. Re-Ranking Algorithm Loaded !!!")
        
    def load_embeddings(self):
        '''
        Load all four vector databases with safe deserialization
        '''
        # Broker-related information
        self.broker_vector_db = FAISS.load_local(
            "../data/vectordata/broker_vector_db", 
            self.sentence_transformer,
            allow_dangerous_deserialization=True
        )
        
        # Fundamental data of stocks
        self.fundamental_vector_db = FAISS.load_local(
            "../data/vectordata/fundamental_vector_db", 
            self.sentence_transformer,
            allow_dangerous_deserialization=True
        )
        
        # Company/stock details from 2008-2025
        self.company_vector_db = FAISS.load_local(
            "../data/vectordata/company_vector_db", 
            self.sentence_transformer,
            allow_dangerous_deserialization=True
        )
        
        # General NEPSE information, rules, IPOs, etc.
        self.pdf_vector_db = FAISS.load_local(
            "../data/vectordata/pdf_vector_db", 
            self.sentence_transformer,
            allow_dangerous_deserialization=True
        )

        # ✅ FIX: Create a dictionary for easier debugging
        self.faiss_stores = {
            "broker_vector_db": self.broker_vector_db,
            "fundamental_vector_db": self.fundamental_vector_db,
            "company_vector_db": self.company_vector_db,
            "pdf_vector_db": self.pdf_vector_db
        }
        print("\nTesting manual FAISS search...")

        try:
            query_embedding = pipeline.sentence_transformer.embed_query("test query")
            print("Query Embedding Shape:", len(query_embedding))
            
            docs_and_scores = pipeline.broker_vector_db.similarity_search_by_vector(query_embedding, k=5)
            print("Retrieved docs:", docs_and_scores)

        except Exception as e:
            print("Manual FAISS search error:", str(e))

        
        # Debugging info for each store
        for store_name, vector_store in self.faiss_stores.items():
            try:
                print(f"Vector store '{store_name}' stats:")
                print(f"- Index size: {vector_store.index.ntotal}")
                try:
                    # Test manual embedding
                    query_embedding = self.sentence_transformer.embed_query("test")
                    print(f"- Test Embedding Shape: {len(query_embedding)}")
                except Exception as e:
                    print(f"- Embedding error: {e}")

            except Exception as e:
                print(f"- Error in store '{store_name}': {str(e)}")
        
        print("5. All FAISS vector stores loaded successfully!!!")


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
        try:
            # Check if question is in Nepali
            is_original_language_nepali = self.is_text_nepali(question)

            # Translate if Nepali
            if is_original_language_nepali:
                try:
                    question = self.perform_translation(question, 'ne', 'en')
                    print("Translated Question: ", question)
                    if isinstance(question, list):
                        yield "data: " + str(question[0])+"\n\n"
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
            
            # Enhanced database-specific prompts
            prompt_templates = {
                "broker": '''
                You are NEPSE-GPT, an expert on Nepalese stock brokers. Answer the question using ONLY the information provided in the context below.
                
                CONTEXT INFORMATION:
                {context}
                
                QUESTION: {question}
                
                BROKER DATABASE STRUCTURE:
                The broker information typically contains these fields in order:
                - ID Number
                - License Number (sometimes with _RWS suffix for branch offices)
                - Company Name 
                - Address (City, District)
                - Phone (may be empty)
                - Website URL
                - TMS (Trading Management System) URL
                
                INSTRUCTIONS:
                1. If answering about a specific broker, include their name, license number, address, and website.
                2. For broker branches, note they share the same license number but with "_RWS" suffix.
                3. When mentioning locations, be precise about which city each broker office is located in.
                4. If asked about brokers in a specific location (e.g., "brokers in Kathmandu"), count only the offices in that exact location.
                5. Count each broker office separately (main office and branches are counted individually).
                6. Understand that the same broker company may have multiple locations (e.g., Agrawal Securities has offices in Dillibazar, Biratnagar, and Janakpur).
                7. If the question asks about total brokers in Nepal, count each unique license number (without _RWS) only once.
                8. If specific information is not in the context, state "Based on the available information, I cannot provide details about [specific item]."
                9. Keep your answer concise (3-5 sentences) but include all relevant facts from the context.
                10. Do not make up or infer information not present in the context.
                
                ANSWER:
                ''',
                
                "fundamental": '''
                You are NEPSE-GPT, a financial analyst specialized in Nepalese stock fundamentals. Analyze the question using ONLY the provided context which contains the LATEST fundamental data (typically last 1-2 quarters). When historical context is missing, clearly state this limitation.

                CONTEXT INFORMATION:
                {context}

                QUESTION: {question}

                RESPONSE PROTOCOL:
                1. DATA PRESENTATION:
                - Lead with the most recent metrics first
                - Format: "[Metric]: [Value] [Fiscal Year/Qtr] (e.g., EPS: Rs. 16.03 FY2081 Q2)"
                - Highlight significant changes with ▲/▼ arrows when comparable data exists
                - Negative values: "WARNING: Negative EPS of Rs. -2.45 (FY2081 Q3)"

                2. HISTORICAL CONTEXT HANDLING:
                - If question asks for historical trends: "The available data only covers [time period]. For complete historical analysis, please consult annual reports."
                - For missing year-end data: "FY2083 annual figures not yet available. Latest data is from Q3."

                3. FINANCIAL BENCHMARKING:
                EPS Assessment (Latest Available):
                - > Rs. 25: "Strong earnings growth"
                - Rs. 15-25: "Healthy profitability"
                - Rs. 5-15: "Moderate performance"
                - < Rs. 5: "Concerning earnings quality" 
                
                P/E Evaluation:
                - Sector Average: ~22 (compare accordingly)
                - < 15: "Undervalued relative to sector"
                - 15-25: "Fairly valued"
                - > 25: "Premium valuation"

                4. DIVIDEND ANALYSIS:
                - Current Yield: "[X]% (based on last market price)"
                - Historical Pattern: "Consistent payer" or "Volatile history" if data permits

                5. RISK INDICATORS:
                - Debt Warning: "High PBV of 4.2 suggests overvaluation"
                - Negative Metrics: "CAUTION: Quarterly EPS decline from Rs. 15 → Rs. 8"

                6. TEMPLATE RESPONSE:
                "[Company] Fundamentals (Latest Available):
                - EPS: Rs. [X] ([Fiscal/Qtr]) ▲[Y]% from previous
                - P/E: [X] vs sector average [Y]
                - Book Value: Rs. [X] (PBV: [Y])
                - Dividend: [X]% for [Fiscal Year]
                - Market Cap: Rs. [X] Cr

                Analysis: [Concise interpretation based on above metrics]. 
                Note: Complete historical trends not available in this dataset."

                ANSWER:
                ''',
                "company": '''
                You are NEPSE-DataAnalyst, providing strictly fact-based analysis of Nepalese stock market data from 2008-2025.

                CONTEXT INFORMATION:
                {context}

                QUESTION: {question}

                RESPONSE PROTOCOL:

                1. DATE AVAILABLE IN DATA:
                """
                [Company] on [Date]:
                - Closing: Rs. [close] (Range: Rs. [min]-Rs. [max])
                - Change: [diff] Rs. ([% change]%)
                - Volume: [tradedShares] shares (Rs. [amount])
                Technical Indicators:
                • SMA: [SMA] ([Above/Below] price)
                • RSI: [RSI] ([Overbought/Oversold/Neutral])
                • Bollinger: [Position relative to bands]
                """

                2. DATE MISSING FROM DATA:
                """
                No trading record found for [date]. Possible scenarios:
                
                A) Market Holiday:
                - Common during Dashain/Tihar (Sep-Nov)
                - Saturday/Sunday closures
                - Other public holidays
                
                B) Non-Trading Day for this Stock:
                - Listing after [date] (first trade: [first_available_date])
                - Trading suspension
                
                Nearest Available Data:
                - Previous: [last_date] ([last_close] Rs.)
                - Next: [next_date] ([next_close] Rs.)
                """

                3. TECHNICAL INTERPRETATION:
                """
                Technical Context:
                - Trend: [Up/Down/Sideways] since [reference_date]
                - Momentum: [RSI analysis]
                - Key Levels: 
                    Support: Rs. [level]
                    Resistance: Rs. [level]
                - Volume Trend: [Increasing/Decreasing]
                """

                4. EXAMPLE RESPONSES:

                A) For existing date:
                """
                NABIL on 2019-03-07:
                - Closing: Rs. 566 (Range: Rs. 555-583)
                - Change: +10 Rs. (+1.8%)
                - Volume: 1,810 shares (Rs. 1,015,090)
                
                Technicals:
                • SMA: 464.2 (Price trading above)
                • RSI: 81.88 (Overbought)
                • Bollinger: Near upper band (709.94)
                
                Analysis:
                - Strong uptrend but overbought
                - Next resistance at Rs. 583
                - Volume decreased 27% from previous day
                """

                B) For missing date:
                """
                No data for 2019-05-29. Possible holiday.
                
                Nearest Sessions:
                - 2019-05-28: Closed at Rs. 890 (RSI: 68)
                - 2019-05-30: Opened at Rs. 900 (+1.2%)
                
                Technical Context:
                - Price was in uptrend before gap
                - Moderate RSI suggests room for movement
                - Watch Rs. 900 as new support level
                """

                RULES:
                1. NEVER invent holiday names
                2. Only reference ACTUAL available dates
                3. For pre-listing queries: "Company listed on [date]"
                4. Always show nearest available data points
                5. Technical analysis only on existing data

                ANSWER:
                ''',
                "general": '''
                You are NEPSE-GPT, an expert on Nepalese stock market rules, procedures, and investor education. Answer the question using ONLY the information provided in the context below.

                CONTEXT INFORMATION:
                {context}

                QUESTION: {question}

                PDF DATABASE STRUCTURE:
                The documents cover a wide range of topics including:
                - SEBON rules, guidelines, and regulations (e.g., Public Issue Regulation, Mutual Fund Regulation, Securities Act)
                - NEPSE operational rules and trading system manuals (e.g., trading hours, circuit breaker rules, listing procedures)
                - CDSC procedures and investor-related services (e.g., demat process, IPO application process, share allotment)
                - Educational content and FAQs (e.g., "What is an IPO?", "Role of a Broker", glossary of stock terms)
                - Government notices, circulars, and updates related to securities and the capital market

                INSTRUCTIONS:
                1. Keep your answer concise (3–5 sentences), clear, and based ONLY on the provided context.
                2. Use bullet points or lists only if the question requests a list or step-by-step explanation.
                3. If the question is about procedures (e.g., IPO process, share transfer), explain them in simple terms.
                4. If the answer relates to regulatory documents (e.g., SEBON rules), mention the name of the document if it's available.
                5. DO NOT make up or assume any information that is not present in the context.
                6. If the context does not contain the requested detail, respond with: "Based on the available information, I cannot provide details about [specific topic]."

                ANSWER:
                '''
            }
            
            # Check for counting queries to use specialized prompt
            query_lower = question.lower()
            counting_patterns = ["how many", "count", "total number", "number of", "list all", "show all", "all the", "count of"]
            is_counting_query = any(pattern in query_lower for pattern in counting_patterns)
            
            if is_counting_query:
                # Add specialized counting prompt that emphasizes accuracy in counting entities
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
                
                # Use counting prompt regardless of DB type for counting queries
                prompt = counting_prompt.format(question=question, context=context)
            else:
                # Select the appropriate standard prompt template
                prompt_template = prompt_templates.get(db_type, prompt_templates["general"])
                prompt = prompt_template.format(question=question, context=context)

            # Generate response with improved error handling
            try:
                inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
                
                # Model generation parameters
                generation_kwargs = dict(
                    inputs, 
                    streamer=self.streamer, 
                    max_new_tokens=2000,       # Maximum number of tokens to generate
                    do_sample=True,            # Enable sampling
                    temperature=0.3,           # Lower temperature for more factual/deterministic responses
                    top_p=0.95,                # Nucleus sampling parameter
                    top_k=40,                  # Top-k sampling parameter
                    repetition_penalty=1.1,    # Discourage repetition
                    pad_token_id=50256         # Padding token ID
                )
                
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                if is_original_language_nepali:
                    # Handle Nepali translation for streaming
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
                                    yield f"data: {sentence}\n\n"  # Fallback to English
                                    sentence = ""
                else:
                    # Stream English response directly
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
            # Global error handler
            print(f"Unexpected error in make_predictions: {e}")
            yield "data: Sorry, an unexpected error occurred. Please try again later.\n\n"
            yield "data: END\n\n"
            return
        
if __name__ == "__main__":
    # Instantiate the pipeline
    pipeline = PredictionPipeline()
    
    # Load everything
    pipeline.load_model_and_tokenizers()
    pipeline.load_sentence_transformer()
    pipeline.load_reranking_model()
    pipeline.load_embeddings()

    # ✅ Multiple test queries mapped to intended vector stores
    test_queries = {
        "broker_vector_db": "Which brokers are located in Kathmandu?",
        "fundamental_vector_db": "What is the EPS of NABIL bank?",
        "company_vector_db": "Show me the stock price history of NLIC in 2020.",
        "pdf_vector_db": "What are the trading rules in NEPSE?"
    }

    # ✅ Loop through and test each
    for db_name, test_question in test_queries.items():
        print(f"\n=== Testing Query for {db_name} ===")
        print(f"Test Question: {test_question}\n")
        
        response = ""
        for output in pipeline.make_predictions(test_question):
            if output.startswith("data: "):
                content = output.replace("data: ", "").strip()
                if content == "END":
                    break
                response += content + " "
        
        print("Final Answer:\n")
        print(response.strip())
        print("\n" + "="*80 + "\n")
