# Try to import all required libraries and handle potential import errors
try:
    # Import transformers for loading language models and tokenizers
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    # Import HuggingFace embeddings for creating vector representations of text
    from langchain.embeddings import HuggingFaceEmbeddings
    # Import FAISS for efficient vector storage and similarity search
    from langchain.vectorstores import FAISS
    # Import FlagModel for reranking search results
    from FlagEmbedding import FlagModel
    # Import requests for making HTTP requests (e.g., for translation API)
    import requests
    # Import re for regular expression operations (e.g., parsing text)
    import re
    # Import urllib.parse for URL encoding (e.g., for translation queries)
    import urllib.parse
    # Import torch for GPU/CPU tensor operations and model handling
    import torch
    # Import Thread for running model generation in a separate thread
    from threading import Thread
    # Import os for file and directory operations
    import os
    # Import shutil for high-level file operations (e.g., moving directories)
    import shutil
    # Import zipfile for handling zip file extraction
    import zipfile
    # Import pandas for data manipulation and analysis
    import pandas as pd
    # Import numpy for numerical operations
    import numpy as np
    # Import technical analysis indicators for stock analysis
    from ta.trend import SMAIndicator, EMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands
    # Import matplotlib for static plotting
    import matplotlib.pyplot as plt
    # Import plotly for interactive candlestick charts
    import plotly.graph_objects as go
    # Import base64 for encoding images to display in HTML
    import base64
    # Import BytesIO for in-memory binary streams (e.g., saving plots)
    from io import BytesIO
    # Import json for parsing JSON data
    import json
    # Print success message if all imports work
    print("All imports succeeded!")
# Catch and display any import errors
except ImportError as e:
    print(f"Import error: {e}")

# Define the PredictionPipeline class to encapsulate the prediction logic
class PredictionPipeline:
    # Initialize the pipeline with model configurations and paths
    def __init__(self):
        # Set the model ID for the pretrained language model
        self.model_id = "TheBloke/neural-chat-7B-v3-1-GPTQ"
        # Set temperature for controlling randomness in text generation
        self.temperature = 0.3
        # Define possible quantization options for the model
        self.bit = ["gptq-4bit-32g-actorder_True", "gptq-8bit-128g-actorder_True"]
        # Specify the sentence transformer model for embeddings
        self.sentence_transformer_modelname = 'sentence-transformers/all-mpnet-base-v2'
        # Determine if GPU is available, else use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Print the device being used for computations
        print(f"1. Device being utilized: {self.device} !!!")
        # Define paths to vector databases for different domains
        self.vector_db_paths = {
            "broker": "/content/broker_vector_db",
            "fundamental": "/content/fundamental_vector_db",
            "pdf": "/content/pdf_vector_db",
            "stock_company": "/content/stock_company_vector_db",
            "stock_date": "/content/stock_date_vector_db"
        }

    # Load the language model and tokenizer
    def load_model_and_tokenizers(self):
        # Initialize the tokenizer with specified model and settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map=self.device, use_fast=True, model_max_length=4000
        )
        # Load the pretrained causal language model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map=self.device, trust_remote_code=False, revision=self.bit[0]
        )
        # Create a streamer for token-by-token text generation
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        # Confirm successful model loading
        print(f"2. {self.model_id} has been successfully loaded !!!")
        # Display GPU memory usage after loading the model
        print(f"GPU memory after model: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    # Load the sentence transformer for generating embeddings
    def load_sentence_transformer(self):
        # Initialize HuggingFace embeddings with the specified model
        self.sentence_transformer = HuggingFaceEmbeddings(
            model_name=self.sentence_transformer_modelname,
            model_kwargs={'device': self.device},
        )
        # Confirm successful loading of the sentence transformer
        print("3. Sentence Transformer Loaded !!!!!!")
        # Display GPU memory usage after loading embeddings
        print(f"GPU memory after embeddings: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    # Load the reranking model for improving search results
    def load_reranking_model(self):
        # Initialize the FlagModel for reranking with FP16 precision
        self.reranker = FlagModel('BAAI/bge-reranker-large', use_fp16=True)
        # Confirm successful loading of the reranker
        print("4. Re-Ranking Algorithm Loaded !!!")
        # Display GPU memory usage after loading the reranker
        print(f"GPU memory after reranker: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    # Load vector databases from uploaded zip files
    def load_embeddings(self):
        # List all zip files in the content directory
        uploaded_files = [f for f in os.listdir('/content') if f.endswith('.zip')]
        # Print the found zip files
        print("Uploaded zip files found:", uploaded_files)
        # Initialize dictionary to store loaded vector databases
        self.vector_dbs = {}
        # Iterate through each domain and its corresponding vector DB path
        for domain, path in self.vector_db_paths.items():
            # Define the expected zip file name for the domain
            zip_file = f"{domain}_vector_db.zip"
            # Check if the zip file exists in uploaded files
            if zip_file in uploaded_files:
                # Print status of unzipping
                print(f"\nUnzipping {zip_file}...")
                # Open and extract the zip file to a temporary directory
                with zipfile.ZipFile(f"/content/{zip_file}", 'r') as zip_ref:
                    zip_ref.extractall("/content/temp_extract")
                    # Get the extracted folder name
                    extracted_folder = os.path.join("/content/temp_extract", os.listdir("/content/temp_extract")[0])
                    # Remove existing vector DB path if it exists
                    if os.path.exists(path):
                        shutil.rmtree(path)
                    # Move extracted folder to the designated path
                    shutil.move(extracted_folder, path)
                    # Clean up temporary extraction directory
                    shutil.rmtree("/content/temp_extract")
                # Print contents of the vector DB directory
                print(f"Files in {path}:", os.listdir(path))
                # Load the FAISS vector store from the extracted path
                self.vector_dbs[domain] = FAISS.load_local(
                    path, self.sentence_transformer, allow_dangerous_deserialization=True
                )
                # Confirm successful loading of the vector store
                print(f"5. FAISS VECTOR STORE LOADED for {domain.upper()} at '{path}' !!!")
            else:
                # Warn if the zip file for the domain is missing
                print(f"WARNING: {zip_file} not uploaded. Skipping {domain} vector store.")

    # Check if the input text is in Nepali using Unicode range
    def is_text_nepali(self, text):
        # Define regex pattern for Nepali Devanagari characters
        nepali_regex = re.compile(r'[\u0900-\u097F]+')
        # Return True if Nepali characters are found
        return bool(nepali_regex.search(text))

    # Translate text using Google Translate API
    def translate_using_google_api(self, text, source_language="auto", target_language="ne", timeout=5):
        # Define regex pattern to extract translated text from response
        pattern = r'(?s)class="(?:t0|result-container)">(.*?)<'
        # URL-encode the input text for the API request
        escaped_text = urllib.parse.quote(text.encode('utf8'))
        # Construct the Google Translate URL
        url = f'https://translate.google.com/m?tl={target_language}&sl={source_language}&q={escaped_text}'
        # Send GET request to the translation API
        response = requests.get(url, timeout=timeout)
        # Decode the response to UTF-8
        result = response.text.encode('utf8').decode('utf8')
        # Extract the translated text using regex
        result = re.findall(pattern, result)
        # Return the translated text or original text if extraction fails
        return result[0] if result else text

    # Handle text translation for long inputs
    def perform_translation(self, question, source_language, target_language):
        # Try to translate the question
        try:
            # Check if the question exceeds Google's character limit
            if len(question) > 5000:
                # Split English text by periods
                if source_language == "en":
                    splitted_text = question.split(".")
                # Split Nepali text by Nepali full stop
                elif source_language == "ne":
                    splitted_text = question.split("।")
                # Split other languages into chunks of 5000 characters
                else:
                    splitted_text = [question[i:i + 5000] for i in range(0, len(question), 5000)]
                # Translate each chunk and join results
                translated_text = " ".join([self.translate_using_google_api(i, source_language, target_language) for i in splitted_text])
                return translated_text
            else:
                # Translate the entire text if under limit
                return self.translate_using_google_api(question, source_language, target_language)
        # Handle translation errors
        except Exception as e:
            return f"An error occurred, [{e}], while working with Google Translation API"

    # Select the appropriate vector database based on the question
    def select_vector_db(self, question):
        # Convert question to lowercase for keyword matching
        question_lower = question.lower()
        # Check for broker-related keywords
        if "broker" in question_lower:
            return "broker"
        # Check for fundamental analysis keywords
        elif any(kw in question_lower for kw in ["eps", "p/e", "book value", "pbv", "market cap", "fundamental", "compare fundamental"]):
            return "fundamental"
        # Check for policy or PDF-related keywords
        elif any(kw in question_lower for kw in ["rule", "regulation", "pdf", "nepse policy"]):
            return "pdf"
        # Check for technical analysis keywords
        elif any(kw in question_lower for kw in ["rsi", "macd", "moving average", "bollinger", "technical analysis", "compare rsi"]):
            return "stock_company"
        # Check for date-specific stock data keywords
        elif any(kw in question_lower for kw in ["date", "daily", "stock price"]):
            return "stock_date"
        # Default to fundamental database
        else:
            return "fundamental"

    # Rerank search results to improve relevance
    def rerank_contexts(self, query, contexts, number_of_reranked_documents_to_select=3):
        # Encode the query into an embedding
        embeddings_1 = self.reranker.encode(query)
        # Encode the contexts into embeddings
        embeddings_2 = self.reranker.encode(contexts)
        # Compute similarity between query and contexts
        similarity = embeddings_1 @ embeddings_2.T
        # Get the number of contexts
        number_of_contexts = len(contexts)
        # Adjust the number of documents to select if necessary
        if number_of_reranked_documents_to_select > number_of_contexts:
            number_of_reranked_documents_to_select = number_of_contexts
        # Get indices of top-ranked contexts
        highest_ranked_indices = sorted(range(len(similarity)), key=lambda i: similarity[i], reverse=True)[:number_of_reranked_documents_to_select]
        # Return the top-ranked contexts
        return [contexts[index] for index in highest_ranked_indices]

    # Format fundamental data into an HTML table with insights
    def format_fundamental_table(self, companies_data):
        # Define thresholds for coloring metrics
        thresholds = {"EPS": 20, "P/E Ratio": 15, "PBV": 1.5, "Market Capitalization": 10000000000}
        # Define meanings for each metric
        meanings = {
            "EPS": "Higher EPS = More Profitability",
            "P/E Ratio": "Low P/E = Potential undervaluation",
            "PBV": "Lower PBV = Stock may be undervalued",
            "Market Capitalization": "Larger cap = More stability"
        }
        # Extract company names from data
        company_names = [data.get("Company", f"Stock {i+1}") for i, data in enumerate(companies_data)]
        # Start building the HTML table
        table = "<table border='1'><tr><th>Metric</th>" + "".join(f"<th>{name}</th>" for name in company_names) + "<th>Meaning</th></tr>"
        # Define metrics to include in the table
        metrics = ["EPS", "P/E Ratio", "PBV", "Market Capitalization"]
        # Collect values for each metric
        metric_values = {metric: [data.get(metric, "N/A") for data in companies_data] for metric in metrics}
        # Build table rows for each metric
        for metric in metrics:
            row = f"<td>{metric}</td>"
            # Process each value for the metric
            for value in metric_values[metric]:
                # Default color is black
                color = "black"
                # Apply color based on thresholds
                if value != "N/A" and metric in thresholds:
                    # Convert value to float if possible
                    val = float(value) if isinstance(value, (int, float, str)) and value.replace(".", "").isdigit() else None
                    if val is not None:
                        # Color EPS based on thresholds
                        if metric == "EPS" and val > thresholds[metric]: color = "green"
                        elif metric == "EPS" and val < 10: color = "red"
                        # Color P/E Ratio based on thresholds
                        elif metric == "P/E Ratio" and val < thresholds[metric]: color = "green"
                        elif metric == "P/E Ratio" and val > 20: color = "red"
                        # Color PBV based on thresholds
                        elif metric == "PBV" and val < thresholds[metric]: color = "green"
                        elif metric == "PBV" and val > 2: color = "red"
                        # Color Market Cap based on thresholds
                        elif metric == "Market Capitalization" and val > thresholds[metric]: color = "green"
                        elif metric == "Market Capitalization" and val < 1000000000: color = "red"
                # Add value to the row with color styling
                row += f"<td style='color:{color}'>{value}</td>"
            # Add the metric's meaning to the row
            row += f"<td>{meanings[metric]}</td>"
            # Add the row to the table
            table += f"<tr>{row}</tr>"
        # Close the table
        table += "</table>"
        # Generate comparison insights
        insights = self.compare_stocks(companies_data, company_names)
        # Return the table and insights
        return table, insights

    # Compare stocks based on fundamental metrics
    def compare_stocks(self, companies_data, company_names):
        # Initialize scores for each company
        scores = {name: 0 for name in company_names}
        # Initialize insights list
        insights = []
        # Extract EPS values
        eps_values = [float(data.get("EPS", 0)) if data.get("EPS", "N/A") != "N/A" else 0 for data in companies_data]
        # Find the company with the highest EPS
        max_eps_idx = eps_values.index(max(eps_values))
        # Increment score for the highest EPS
        scores[company_names[max_eps_idx]] += 1
        # Add insight about EPS
        insights.append(f"{company_names[max_eps_idx]} has the highest EPS ({eps_values[max_eps_idx]}), indicating strong profitability.")
        # Extract P/E Ratio values
        pe_values = [float(data.get("P/E Ratio", 100)) if data.get("P/E Ratio", "N/A") != "N/A" else 100 for data in companies_data]
        # Find the company with the lowest P/E
        min_pe_idx = pe_values.index(min(pe_values))
        # Check if P/E is favorable
        if pe_values[min_pe_idx] < 15:
            # Increment score for low P/E
            scores[company_names[min_pe_idx]] += 1
            # Add insight about P/E
            insights.append(f"{company_names[min_pe_idx]} has the lowest P/E ({pe_values[min_pe_idx]}), suggesting potential undervaluation.")
        # Extract PBV values
        pbv_values = [float(data.get("PBV", 10)) if data.get("PBV", "N/A") != "N/A" else 10 for data in companies_data]
        # Find the company with the lowest PBV
        min_pbv_idx = pbv_values.index(min(pbv_values))
        # Check if PBV is favorable
        if pbv_values[min_pbv_idx] < 1.5:
            # Increment score for low PBV
            scores[company_names[min_pbv_idx]] += 1
            # Add insight about PBV
            insights.append(f"{company_names[min_pbv_idx]} has the lowest PBV ({pbv_values[min_pbv_idx]}), indicating it may be undervalued.")
        # Extract Market Cap values
        mcap_values = [float(data.get("Market Capitalization", 0)) if data.get("Market Capitalization", "N/A") != "N/A" else 0 for data in companies_data]
        # Find the company with the highest Market Cap
        max_mcap_idx = mcap_values.index(max(mcap_values))
        # Increment score for high Market Cap
        scores[company_names[max_mcap_idx]] += 1
        # Add insight about Market Cap
        insights.append(f"{company_names[max_mcap_idx]} has the largest market cap ({mcap_values[max_mcap_idx]}), suggesting greater stability.")
        # Determine the strongest stock
        strongest_stock = max(scores, key=scores.get)
        # Add final insight about the strongest stock
        insights.append(f"Based on fundamentals, {strongest_stock} appears strongest for investment with a score of {scores[strongest_stock]}/4.")
        # Join insights into a single string
        return " ".join(insights)

    # Parse stock company data into a DataFrame
    def parse_stock_company_data(self, context):
        # Initialize dictionary for OHLCV data
        df_data = {"Date": [], "Open": [], "High": [], "Low": [], "Close": [], "Volume": []}
        # Default stock name
        stock_name = "Unknown Stock"
        # Process each document in the context
        for doc in context:
            try:
                # Parse document as JSON if it's a string
                data = json.loads(doc) if isinstance(doc, str) else doc
                # Extract stock name from metadata if available
                stock_name = getattr(doc, 'metadata', {}).get("company", stock_name) if hasattr(doc, 'metadata') else stock_name
                # Check if data is a dictionary
                if isinstance(data, dict):
                    # Iterate through date entries
                    for date, info in data.items():
                        # Get price information
                        price = info.get("price", {})
                        # Append data to the DataFrame dictionary
                        df_data["Date"].append(date)
                        df_data["Open"].append(price.get("prevClose", 0))
                        df_data["High"].append(price.get("max", 0))
                        df_data["Low"].append(price.get("min", 0))
                        df_data["Close"].append(price.get("close", 0))
                        df_data["Volume"].append(info.get("tradedShares", 0))
                    # Break if data is found
                    if df_data["Date"]:
                        break
            # Handle parsing errors
            except Exception as e:
                print(f"Error parsing stock_company data: {e}")
                continue
        # Check if any data was collected
        if not df_data["Date"]:
            return None, stock_name, "No valid OHLCV data found in stock_company context."
        # Create DataFrame from collected data
        df = pd.DataFrame(df_data)
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Drop rows with invalid dates and sort by date
        df = df.dropna(subset=['Date']).sort_values('Date').set_index('Date')
        # Return the DataFrame, stock name, and no error
        return df, stock_name, None

    # Parse stock date data into a DataFrame
    def parse_stock_date_data(self, context, company_name=None):
        # Initialize dictionary for OHLCV data
        df_data = {"Date": [], "Open": [], "High": [], "Low": [], "Close": [], "Volume": []}
        # Set stock name based on input or default
        stock_name = company_name if company_name else "Multiple Stocks"
        # Process each document in the context
        for doc in context:
            try:
                # Parse document as JSON if it's a string
                data = json.loads(doc) if isinstance(doc, str) else doc
                # Extract date from metadata if available
                date = getattr(doc, 'metadata', {}).get("date", "Unknown Date") if hasattr(doc, 'metadata')扁0
                # Iterate through company data
                for company_data in data.get("data", []):
                    # Get company information
                    comp = company_data.get("company", {})
                    # Skip if company name doesn't match
                    if company_name and comp.get("name") != company_name:
                        continue
                    # Get price information
                    price = company_data.get("price", {})
                    # Append data to the DataFrame dictionary
                    df_data["Date"].append(date)
                    df_data["Open"].append(price.get("prevClose", 0))
                    df_data["High"].append(price.get("max", 0))
                    df_data["Low"].append(price.get("min", 0))
                    df_data["Close"].append(price.get("close", 0))
                    df_data["Volume"].append(company_data.get("tradedShares", 0))
                    # Update stock name if not specified
                    if not company_name:
                        stock_name = comp.get("name", stock_name)
            # Handle parsing errors
            except Exception as e:
                print(f"Error parsing stock_date data: {e}")
                continue
        # Check if any data was collected
        if not df_data["Date"]:
            return None, stock_name, "No valid data found in stock_date context."
        # Create DataFrame from collected data
        df = pd.DataFrame(df_data)
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Drop rows with invalid dates and sort by date
        df = df.dropna(subset=['Date']).sort_values('Date').set_index('Date')
        # Return the DataFrame, stock name, and no error
        return df, stock_name, None

    # Calculate technical indicators for the given DataFrame
    def calculate_technical_indicators(self, df):
        # Define required columns for technical analysis
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Check if all required columns exist and DataFrame is not empty
        if not all(col in df.columns for col in required_cols) or df.empty:
            return None, "Insufficient OHLCV data for technical analysis."
        # Convert columns to numeric, handling errors
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Check for missing values in required columns
        if df[required_cols].isna().all().any():
            return None, "Missing values in OHLCV data."
        # Initialize dictionary for indicators
        indicators = {}
        # Calculate Simple Moving Average (20-day)
        indicators['SMA'] = SMAIndicator(df['Close'], window=20).sma_indicator()
        # Calculate Exponential Moving Average (20-day)
        indicators['EMA'] = EMAIndicator(df['Close'], window=20).ema_indicator()
        # Calculate Relative Strength Index (14-day)
        indicators['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        # Calculate MACD (12, 26, 9)
        macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        indicators['MACD'] = macd.macd()
        indicators['MACD_Signal'] = macd.macd_signal()
        indicators['MACD_Hist'] = macd.macd_diff()
        # Calculate Bollinger Bands (20-day, 2 std dev)
        bb = BollingerBands(df['Close'], window=20, window_dev=2)
        indicators['BB_High'] = bb.bollinger_hband()
        indicators['BB_Low'] = bb.bollinger_lband()
        indicators['BB_Mid'] = bb.bollinger_mavg()
        # Return indicators and no error
        return indicators, None

    # Generate a chart for technical indicators
    def generate_technical_chart(self, df, indicators, indicator_type=None, stock_name="Stock"):
        # Generate RSI chart if specified
        if indicator_type == 'RSI':
            # Create a new figure for RSI
            fig, ax = plt.subplots(figsize=(10, 4))
            # Plot RSI values
            ax.plot(df.index, indicators['RSI'], label='RSI', color='purple')
            # Add overbought line at 70
            ax.axhline(70, color='red', linestyle='--', label='Overbought (70)')
            # Add oversold line at 30
            ax.axhline(30, color='green', linestyle='--', label='Oversold (30)')
            # Set chart title
            ax.set_title(f'RSI for {stock_name}')
            # Set x-axis label
            ax.set_xlabel('Date')
            # Set y-axis label
            ax.set_ylabel('RSI')
            # Add legend
            ax.legend()
        # Generate MACD chart if specified
        elif indicator_type == 'MACD':
            # Create a new figure for MACD
            fig, ax = plt.subplots(figsize=(10, 4))
            # Plot MACD line
            ax.plot(df.index, indicators['MACD'], label='MACD', color='blue')
            # Plot Signal line
            ax.plot(df.index, indicators['MACD_Signal'], label='Signal', color='orange')
            # Plot MACD Histogram
            ax.bar(df.index, indicators['MACD_Hist'], label='Histogram', color='gray', alpha=0.5)
            # Set chart title
            ax.set_title(f'MACD for {stock_name}')
            # Set x-axis label
            ax.set_xlabel('Date')
            # Set y-axis label
            ax.set_ylabel('MACD')
            # Add legend
            ax.legend()
        # Generate candlestick chart with SMA and Bollinger Bands
        else:
            # Create a new Plotly figure
            fig = go.Figure()
            # Add candlestick chart
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'))
            # Add SMA line
            fig.add_trace(go.Scatter(x=df.index, y=indicators['SMA'], line=dict(color='blue'), name='SMA 20'))
            # Add Bollinger Band upper line
            fig.add_trace(go.Scatter(x=df.index, y=indicators['BB_High'], line=dict(color='red', dash='dash'), name='BB Upper'))
            # Add Bollinger Band lower line
            fig.add_trace(go.Scatter(x=df.index, y=indicators['BB_Low'], line=dict(color='green', dash='dash'), name='BB Lower'))
            # Update layout with title and labels
            fig.update_layout(title=f'Technical Analysis for {stock_name}', xaxis_title='Date', yaxis_title='Price', template='plotly_white')
        # Create a BytesIO buffer for the image
        buf = BytesIO()
        # Save matplotlib figure to buffer for RSI/MACD
        if indicator_type in ['RSI', 'MACD']:
            # Adjust layout to prevent clipping
            plt.tight_layout()
            # Save figure to buffer
            plt.savefig(buf, format='png')
            # Close the figure to free memory
            plt.close(fig)
        # Save Plotly figure to buffer for candlestick
        else:
            # Write Plotly figure to buffer
            fig.write_image(buf, format='png')
        # Reset buffer position
        buf.seek(0)
        # Encode image to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        # Return HTML image tag with base64 data
        return f"<img src='data:image/png;base64,{img_base64}'/>"

    # Generate a comparison chart for RSI
    def generate_comparison_chart(self, dfs, indicators_list, indicator_type, stock_names):
        # Generate RSI comparison chart
        if indicator_type == 'RSI':
            # Create a new figure for RSI comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            # Plot RSI for each stock
            for df, indicators, name in zip(dfs, indicators_list, stock_names):
                ax.plot(df.index, indicators['RSI'], label=f'RSI - {name}')
            # Add overbought line at 70
            ax.axhline(70, color='red', linestyle='--', label='Overbought (70)')
            # Add oversold line at 30
            ax.axhline(30, color='green', linestyle='--', label='Oversold (30)')
            # Set chart title
            ax.set_title(f'RSI Comparison: {", ".join(stock_names)}')
            # Set x-axis label
            ax.set_xlabel('Date')
            # Set y-axis label
            ax.set_ylabel('RSI')
            # Add legend
            ax.legend()
        # Create a BytesIO buffer for the image
        buf = BytesIO()
        # Adjust layout to prevent clipping
        plt.tight_layout()
        # Save figure to buffer
        plt.savefig(buf, format='png')
        # Close the figure to free memory
        plt.close(fig)
        # Reset buffer position
        buf.seek(0)
        # Encode image to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        # Return HTML image tag with base64 data
        return f"<img src='data:image/png;base64,{img_base64}'/>"

    # Analyze technical indicators and provide insights
    def analyze_technical_indicators(self, indicators, stock_name):
        # Get the latest RSI value
        latest_rsi = indicators['RSI'].iloc[-1]
        # Get the latest MACD value
        latest_macd = indicators['MACD'].iloc[-1]
        # Get the latest MACD Signal value
        latest_signal = indicators['MACD_Signal'].iloc[-1]
        # Get the latest closing price (using BB Mid)
        latest_close = indicators['BB_Mid'].iloc[-1]
        # Initialize insights list
        insights = []
        # Analyze RSI
        if latest_rsi > 70:
            insights.append(f"{stock_name}'s RSI ({latest_rsi:.2f}) is above 70, indicating overbought conditions.")
        elif latest_rsi < 30:
            insights.append(f"{stock_name}'s RSI ({latest_rsi:.2f}) is below 30, suggesting oversold conditions.")
        # Analyze MACD
        if latest_macd > latest_signal:
            insights.append(f"{stock_name}'s MACD is above the signal line, suggesting bullish momentum.")
        elif latest_macd < latest_signal:
            insights.append(f"{stock_name}'s MACD is below the signal line, indicating bearish momentum.")
        # Analyze Bollinger Bands
        if latest_close > indicators['BB_High'].iloc[-1]:
            insights.append(f"{stock_name}'s price is above the Bollinger Band upper limit, signaling potential overextension.")
        elif latest_close < indicators['BB_Low'].iloc[-1]:
            insights.append(f"{stock_name}'s price is below the Bollinger Band lower limit, suggesting a possible rebound.")
        # Return insights or a default message
        return " ".join(insights) if insights else f"No significant technical signals detected for {stock_name}."

    # Compare technical indicators between stocks
    def compare_technical_indicators(self, indicators_list, stock_names):
        # Initialize insights list
        insights = []
        # Get latest RSI values for each stock
        rsi_values = [indicators['RSI'].iloc[-1] for indicators in indicators_list]
        # Analyze RSI for each stock
        for name, rsi in zip(stock_names, rsi_values):
            if rsi > 70:
                insights.append(f"{name}'s RSI ({rsi:.2f}) is above 70, indicating overbought conditions.")
            elif rsi < 30:
                insights.append(f"{name}'s RSI ({rsi:.2f}) is below 30, suggesting oversold conditions.")
            else:
                insights.append(f"{name}'s RSI ({rsi:.2f}) is neutral.")
        # Compare RSI values
        if all(30 <= rsi <= 70 for rsi in rsi_values):
            insights.append("Both stocks have neutral RSI values; no clear preference based on RSI alone.")
        elif rsi_values[0] > rsi_values[1]:
            insights.append(f"{stock_names[0]} has a higher RSI, suggesting stronger momentum but possible overbought risk compared to {stock_names[1]}.")
        else:
            insights.append(f"{stock_names[1]} has a higher RSI, suggesting stronger momentum but possible overbought risk compared to {stock_names[0]}.")
        # Join insights into a single string
        return " ".join(insights)

    # Main function to process queries and generate predictions
    def make_predictions(self, question, top_n_values=10):
        # Check if the question is in Nepali
        is_original_language_nepali = self.is_text_nepali(question)
        # Translate Nepali questions to English
        if is_original_language_nepali:
            question = self.perform_translation(question, 'ne', 'en')
            # Print the translated question
            print("Translated Question: ", question)

        # Select the appropriate vector database
        domain = self.select_vector_db(question)
        # Check if the vector database exists
        if domain not in self.vector_dbs:
            return f"Vector store for {domain} not loaded. Please upload {domain}_vector_db.zip."

        # Get the vector database for the selected domain
        vector_db = self.vector_dbs[domain]
        # Print the selected vector store
        print(f"Using {domain.upper()} vector store for query.")
        # Perform similarity search with scoring
        similarity_search = vector_db.similarity_search_with_score(question, k=top_n_values)
        # Filter contexts based on similarity score
        context = [doc.page_content for doc, score in similarity_search if score < 1.5]
        # Return error if no relevant contexts found
        if not context:
            return "The question asked and domain knowledge provided are irrelevant. Unable to provide an answer to this question."

        # Rerank contexts if multiple are found
        if len(context) > 1:
            context = self.rerank_contexts(question, context)

        # Handle fundamental analysis comparison
        if domain == "fundamental" and "compare" in question.lower():
            # Initialize list for company data
            companies_data = []
            # Process each context document
            for doc in context:
                try:
                    # Parse document as JSON
                    data = json.loads(doc) if isinstance(doc, str) else doc
                    # Check for relevant fundamental metrics
                    if any(key in data for key in ["EPS", "P/E Ratio", "PBV", "Market Capitalization"]):
                        companies_data.append(data)
                # Handle parsing errors
                except Exception as e:
                    print(f"Error parsing fundamental data: {e}")
                    continue
            # Generate table and insights if enough data
            if len(companies_data) >= 2:
                table, insights = self.format_fundamental_table(companies_data[:3])
                return f"{table} {insights}"
            else:
                return "Insufficient data to compare multiple stocks based on fundamentals."

        # Handle technical analysis and comparisons
        if any(kw in question.lower() for kw in ["technical analysis", "rsi", "macd", "moving average", "bollinger"]):
            # Handle RSI comparison
            if "compare" in question.lower() and "rsi" in question.lower():
                # Extract stock names from the question
                stock_names = [company for company in ["AHL", "AKJCL", "ADBL"] if company in question.upper()][:2]
                # Validate number of stocks
                if len(stock_names) != 2:
                    return "Please specify exactly two stocks to compare RSI (e.g., 'Compare RSI of AHL and AKJCL')."
                # Initialize lists for DataFrames and indicators
                dfs, indicators_list, errors = [], [], []
                # Process each stock
                for stock in stock_names:
                    # Search for historical data
                    stock_context = self.vector_dbs["stock_company"].similarity_search_with_score(f"{stock} historical data", k=1)
                    # Parse stock data
                    df, stock_name, error = self.parse_stock_company_data([doc.page_content for doc, score in stock_context if score < 1.5])
                    # Check for errors
                    if error or df is None:
                        errors.append(f"{stock}: {error}")
                        break
                    # Calculate technical indicators
                    indicators, error = self.calculate_technical_indicators(df)
                    # Check for errors
                    if error:
                        errors.append(f"{stock}: {error}")
                        break
                    # Append data
                    dfs.append(df)
                    indicators_list.append(indicators)
                # Return errors if any
                if errors:
                    return f"Errors encountered: {'; '.join(errors)}"
                # Generate RSI comparison chart
                chart = self.generate_comparison_chart(dfs, indicators_list, "RSI", stock_names)
                # Compare technical indicators
                insights = self.compare_technical_indicators(indicators_list, stock_names)
                # Return chart and insights
                return f"{chart} RSI Comparison - {stock_names[0]}: {indicators_list[0]['RSI'].iloc[-1]:.2f}, {stock_names[1]}: {indicators_list[1]['RSI'].iloc[-1]:.2f}. {insights}"

            # Initialize variables for technical analysis
            df, stock_name, error = None, "Unknown Stock", None
            # Parse data based on domain
            if domain == "stock_company":
                df, stock_name, error = self.parse_stock_company_data(context)
            elif domain == "stock_date":
                # Extract company name if specified
                company_name = next((company for company in ["AHL", "AKJCL", "ADBL"] if company in question.upper()), None)
                df, stock_name, error = self.parse_stock_date_data(context, company_name)
            # Check for errors or invalid data
            if error or df is None or df.empty:
                return error or "Insufficient or invalid historical data for technical analysis."
            # Calculate technical indicators
            indicators, error = self.calculate_technical_indicators(df)
            # Check for errors
            if error:
                return error
            # Generate RSI chart and insights
            if "rsi" in question.lower():
                chart = self.generate_technical_chart(df, indicators, "RSI", stock_name)
                insights = self.analyze_technical_indicators(indicators, stock_name)
                return f"{chart} RSI for {stock_name} is {indicators['RSI'].iloc[-1]:.2f}. {insights}"
            # Generate MACD chart and insights
            elif "macd" in question.lower():
                chart = self.generate_technical_chart(df, indicators, "MACD", stock_name)
                insights = self.analyze_technical_indicators(indicators, stock_name)
                return f"{chart} MACD for {stock_name} is {indicators['MACD'].iloc[-1]:.2f}, Signal: {indicators['MACD_Signal'].iloc[-1]:.2f}. {insights}"
            # Generate full technical analysis chart
            else:
                chart = self.generate_technical_chart(df, indicators, None, stock_name)
                insights = self.analyze_technical_indicators(indicators, stock_name)
                return f"Technical Indicators for {stock_name}: SMA 20 is {indicators['SMA'].iloc[-1]:.2f}, RSI is {indicators['RSI'].iloc[-1]:.2f}, MACD is {indicators['MACD'].iloc[-1]:.2f} with Signal {indicators['MACD_Signal'].iloc[-1]:.2f}, Bollinger Bands are High {indicators['BB_High'].iloc[-1]:.2f} and Low {indicators['BB_Low'].iloc[-1]:.2f}. {chart} {insights}"

        # Default behavior: generate answer using the language model
        context = ". ".join(context)
        # Define the prompt for the language model
        prompt = f'''
            Based solely on the information given in the context above, answer the following question.
            Never answer a question in your own words outside of the context provided.
            If the information isn’t available in the context to formulate an answer, politely say "Sorry, I don’t have knowledge about that topic."
            Please do not provide additional explanations or information by answering outside of the context.
            Always answer in maximum five sentences and less than hundred words.
            \n\n
            Question: {question}\n\n
            Context: {context}\n\n
            Answer:
        '''
        # Tokenize the prompt
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        # Define generation parameters
        generation_kwargs = dict(
            inputs, streamer=self.streamer, max_new_tokens=2000, do_sample=True,
            temperature=0.3, top_p=0.95, top_k=40, repetition_penalty=1.1, pad_token_id=50256
        )
        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        # Collect generated tokens
        response = ""
        for token in self.streamer:
            if token != "</s>":
                response += token
        # Wait for the thread to finish
        thread.join()
        # Translate response back to Nepali if needed
        if is_original_language_nepali:
            response = self.perform_translation(response, "en", "ne")
        # Return the stripped response
        return response.strip()

# Instantiate the prediction pipeline
pipeline = PredictionPipeline()
# Load the model and tokenizers
pipeline.load_model_and_tokenizers()
# Load the sentence transformer
pipeline.load_sentence_transformer()
# Load the reranking model
pipeline.load_reranking_model()
# Load the vector databases
pipeline.load_embeddings()

# Define test queries
queries = [
    "Fundamental analysis of Nabil Bank",
    "Compare fundamental analysis of AHL and AKJCL",
    "What is the RSI of AKJCL?",
    "What is the MACD of AHL?",
    "Technical analysis of ADBL",
    "Compare RSI of AHL and AKJCL",
    "What is the stock price of AHL? (in Nepali: एएचएलको स्टक मूल्य कति हो?)",
    "What are the broker rules?",
    "What is the NEPSE policy in the PDF?",
    "Stock price of ADBL on 2008-01-07",
    "What is the weather like today?"
]

# Process each query and print results
for query in queries:
    print(f"\nQuery: {query}")
    response = pipeline.make_predictions(query)
    print(response)
