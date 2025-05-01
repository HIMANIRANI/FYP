import base64
import difflib
import json
import logging
import re
import time
import urllib.parse
from datetime import datetime
from difflib import get_close_matches
from io import BytesIO
from pathlib import Path
from statistics import mean
from threading import Thread
from typing import Dict, List, Optional, Tuple

import faiss
import langdetect
import numpy as np
import pandas as pd
import requests
import torch
from FlagEmbedding import FlagModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)

# Set up logging
logging.basicConfig(level=logging.INFO) # to confirm that things are working as expected
logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self):
        """Initialize the PredictionPipeline with model and data paths."""
        self.model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"
        self.temperature = 0.2
        self.bit = ["main"]
        self.sentence_transformer_modelname = 'sentence-transformers/all-mpnet-base-v2'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reranker_modelname = 'BAAI/bge-reranker-large'
        self.VALID_TICKERS = self.load_valid_tickers()
        self.routing_rules = [
            {"keywords": ["compare", "fundamental"], "min_tickers": 2, "function": "format_fundamental_comparison"},
            {"keywords": ["fundamental"], "min_tickers": 1, "function": "format_fundamental_table"},
            {"keywords": ["compare"], "min_tickers": 2, "function": "get_stock_comparison"},
            {"keywords": ["monthly", "summary"], "min_tickers": 1, "function": "get_monthly_performance_summary"},
            {"keywords": ["weekly"], "min_tickers": 1, "function": "weekly_summary"},
            {"keywords": ["volatility", "risk"], "min_tickers": 1, "function": "get_volatility_analysis"},
            {"keywords": ["technical"], "min_tickers": 1, "function": "get_technical_indicator_summary"},
            {"keywords": ["trend"], "min_tickers": 1, "function": "get_trend_analysis"},
            {"keywords": ["daily", "today"], "min_tickers": 1, "function": "get_daily_summary"},
        ]

        logger.info(f"1. Device being utilized: {self.device}")
        # Define absolute vector store paths
        base_dir = Path(__file__).resolve().parent.parent
        vector_data_dir = base_dir / "data" / "vector data"
        self.vector_db_paths = {
            "broker": vector_data_dir / "broker_vector_db",
            "fundamental": vector_data_dir / "fundamental_vector_db",
            "pdf": vector_data_dir / "pdf_vector_db",
            "company": vector_data_dir / "company_vector_db"
        }
        self.vector_dbs = {}

    def fuzzy_match_ticker(self, word):
        matches = get_close_matches(word.upper(), self.VALID_TICKERS, n=1, cutoff=0.8)
        return matches[0] if matches else None

    def classify_intent_with_llm(self, query: str) -> str:
        prompt = f"Classify this input into one of: [trend, comparison, volatility, fundamentals, technical, daily].\nInput: {query}"
        return self.get_llm_analysis(prompt).strip().lower()


    def detect_language(self, text: str) -> str:
        """
        Detects the language code of the input text.
        Returns: 'en', 'ne', or full ISO 639-1 language code like 'es', 'fr', etc.
        """
        try:
            lang = detect(text)
            return lang
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "unknown"
    
    def load_valid_tickers(self) -> set:
        """Dynamically load ticker symbols from JSON files in updated_company_data"""
        try:
            base_path = Path(__file__).parent.parent
            data_dir = base_path / "data" / "updated_company_data"
            
            if not data_dir.exists():
                logger.error(f"Directory not found: {data_dir}")
                return set()

            # Get all .json files and extract ticker names (uppercase)
            return {
                file.stem.upper()  # Remove .json extension and normalize case
                for file in data_dir.glob("*.json")
                if file.is_file()
            }
        except Exception as e:
            logger.error(f"Error loading valid tickers: {e}")
            return set() 
        
    def load_model_and_tokenizers(self) -> None:
        """Load the language model and tokenizer."""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                trust_remote_code=True,
                revision=self.bit[0]
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                use_fast=True,
                trust_remote_code=True,
                revision=self.bit[0],
                timeout= 50.0
            )
            self.streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                skip_eos_token=True,
                decode_kwargs={"clean_up_tokenization_spaces": True},
                buffer_size=1
            )
            logger.info(f"2. {self.model_id} has been successfully loaded")
            # logger.debug(f"GPU memory after model: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
        except Exception as e:
            logger.error(f"Error loading model and tokenizer: {e}")
            raise

    def load_sentence_transformer(self) -> None:
        """Load the sentence transformer model."""
        try:
            # Check for model name early
            if not self.sentence_transformer_modelname:
                raise ValueError("Sentence transformer model name is not set.")

            logger.info(f"Loading sentence transformer: {self.sentence_transformer_modelname}")
            
            # Load model
            self.sentence_transformer = HuggingFaceEmbeddings(
                model_name=self.sentence_transformer_modelname,
                model_kwargs={'device': self.device},
            )
            
            logger.info(f"3. Sentence Transformer '{self.sentence_transformer_modelname}' loaded on {self.device}")
            
            # Log memory (only if CUDA is available)
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024**3
                logger.debug(f"GPU memory used after embedding model load: {mem:.2f} GiB")
                
        except Exception as e:
            logger.exception("Error while loading the sentence transformer model.")
            raise


    def load_reranking_model(self) -> None:
        """Load the reranking model."""
        try:
            use_fp16 = torch.cuda.is_available()
            logger.info(f"Loading reranker '{self.reranker_modelname}' (use_fp16={use_fp16})")
            self.reranker = FlagModel(
                self.reranker_modelname,
                use_fp16=use_fp16
            )
            logger.info("4. Re-Ranking Algorithm Loaded Successfully")
        except Exception:
            logger.exception("Error loading reranking model")
            raise


    def load_embeddings(self):
        
        def _load_domain(domain: str, db_path: Path):
            logger.info(f"Loading vector store for '{domain}' from {db_path}")
            if not db_path.exists():
                logger.warning(f"Directory missing: {db_path}")
                return

            try:
                files = list(db_path.iterdir())
                if not files:
                    logger.warning(f"No files in: {db_path}")
                    return
            except Exception as e:
                logger.error(f"Cannot list {db_path}: {e}")
                return

            index_file = db_path / "index.faiss"
            if not index_file.exists():
                logger.warning(f"index.faiss not found in {db_path}")
                return

            try:
                vec_db = FAISS.load_local(
                    folder_path=str(db_path),
                    embeddings=self.sentence_transformer,
                    allow_dangerous_deserialization=True
                )

                # Warm-up search
                try:
                    dummy_vec = np.zeros((1, vec_db.index.d), dtype='float32')
                    vec_db.index.search(dummy_vec, 1)
                    logger.debug(f"Warm-up search done for {domain}")
                except Exception as we:
                    logger.warning(f"Warm-up search failed for {domain}: {we}")

                # Metadata logging
                try:
                    count = vec_db.index.ntotal
                    dim = vec_db.index.d
                    logger.info(f"Index '{domain}': dimension={dim}, entries={count}")
                except Exception:
                    pass

                self.vector_dbs[domain] = vec_db
                logger.info(f"Loaded FAISS store for '{domain}' successfully")
            except Exception as e:
                logger.error(f"Failed loading {domain}: {e}")

        # Load each domain in parallel
        threads = []
        for domain, path in self.vector_db_paths.items():
            t = Thread(target=_load_domain, args=(domain, path))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        logger.info("All vector stores loaded successfully")


    def is_text_nepali(self, text: str) -> bool:
        """Detect if text contains Nepali characters."""
        nepali_regex = re.compile(r'[\u0900-\u097F]+')
        return bool(nepali_regex.search(text))

    def translate_using_google_api(self, text: str, source_language: str = "auto", target_language: str = "ne", timeout: int = 5) -> str:
        """Translate text using Google Translate API."""
        pattern = r'(?s)class="(?:t0|result-container)">(.*?)<'
        escaped_text = urllib.parse.quote(text.encode('utf8'))
        url = f'https://translate.google.com/m?tl={target_language}&sl={source_language}&q={escaped_text}'
        try:
            response = requests.get(url, timeout=timeout)
            result = response.text.encode('utf8').decode('utf8')
            matches = re.findall(pattern, result)
            return matches[0] if matches else text
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text

    def perform_translation(self, question: str, source_language: str, target_language: str) -> str:
        """Handle translation for long texts by splitting if necessary."""
        try:
            if len(question) > 5000:
                if source_language == "en":
                    splitted_text = question.split(".")
                elif source_language == "ne":
                    splitted_text = question.split("।")
                else:
                    splitted_text = [question[i:i + 5000] for i in range(0, len(question), 5000)]
                translated_text = " ".join([self.translate_using_google_api(i, source_language, target_language) for i in splitted_text])
                return translated_text
            return self.translate_using_google_api(question, source_language, target_language)
        except Exception as e:
            logger.error(f"Error in perform_translation: {e}")
            return f"An error occurred, [{e}], while working with Google Translation API"

    def select_vector_db(self, question: str) -> str:
        """Pick the best vector DB, with a few hard‐coded overrides and then a semantic fallback."""
        question_lower = question.lower()

        # 1) “Hard” keyword overrides
        if "broker" in question_lower:
            return "broker"
        if "fundamental" in question_lower or any(kw in question_lower for kw in ["eps", "p/e", "book value", "pbv", "market cap"]):
            return "fundamental"
        if "nepse" in question_lower or any(kw in question_lower for kw in ["rule", "regulation", "policy", "pdf"]):
            return "pdf"
        if any(kw in question_lower for kw in ["rsi", "macd", "moving average", "bollinger", "technical analysis", "stock price", "daily"]):
            return "company"

        # 2) If embeddings aren’t loaded yet, default
        if not self.vector_dbs:
            return "pdf"

        # 3) Semantic fallback: pick the domain whose top-1 distance is smallest
        q_emb = np.array(self.sentence_transformer.embed_query(question), dtype="float32").reshape(1, -1)
        best_domain, best_dist = None, float("inf")

        for domain, vec_db in self.vector_dbs.items():
            try:
                D, _ = vec_db.index.search(q_emb, k=1)
                dist = float(D[0][0])
                if dist < best_dist:
                    best_dist, best_domain = dist, domain
            except Exception:
                continue

        # 4) If the best semantic match is still very far, default
        if best_domain is None or best_dist > 0.1:
            return "pdf"

        return best_domain


    def rerank_contexts(
        self,
        query: str,
        contexts: List[str],
        k: int = 3
    ) -> List[str]:
        """Rerank contexts by cosine similarity to the query, return top-k."""
        if not contexts:
            return []

        # 1) Encode without gradients
        with torch.inference_mode():
            q_vec = self.reranker.encode(query)              # shape (d,)
            c_vecs = self.reranker.encode(contexts)          # shape (n, d)

        # 2) Normalize for cosine similarity
        q_vec = q_vec / q_vec.norm()
        c_vecs = c_vecs / c_vecs.norm(dim=1, keepdim=True)

        # 3) Compute similarities and select top-k
        sims = (c_vecs @ q_vec).cpu().numpy()                # shape (n,)
        topk = np.argsort(-sims)[:min(k, len(sims))]        # descending order

        # 4) Optionally log scores
        for idx in topk:
            logger.debug(f"Context[{idx}] score: {sims[idx]:.4f}")

        # 5) Return the top contexts
        return [contexts[i] for i in topk]


    def generate_next_steps(self, company_name: str, company_type: Optional[str] = None) -> str:
        """Generate follow-up suggestions based on company type."""
        options = ["👉 Show historical performance or technical indicators?"]
        if company_type:
            company_type = company_type.lower()
            if company_type in ["bank", "commercial bank", "development bank"]:
                options.insert(0, f"👉 Compare {company_name} with other banks (commercial or development)?")
            elif company_type in ["hydro", "hydropower"]:
                options.insert(0, f"👉 Compare {company_name} with other hydropower companies?")
            elif company_type in ["mutual fund", "investment fund"]:
                options.insert(0, f"👉 Compare {company_name} with other mutual or investment funds?")
            elif company_type in ["insurance", "life insurance", "non-life insurance"]:
                options.insert(0, f"👉 Compare {company_name} with other insurance companies?")
            elif company_type in ["finance", "microfinance"]:
                options.insert(0, f"👉 Compare {company_name} with other finance or microfinance companies?")
        return "<br><br>Would you like me to:<br>" + "<br>".join(options)

    def generate_labels(self, data: Dict) -> str:
        """Generate labels for earnings, valuation, and risk based on fundamental metrics."""
        try:
            eps = float(data.get("EPS", 0))
            pe_ratio = float(data.get("P/E Ratio", 0))
            pbv = float(data.get("PBV", 0))

            # Earnings Label
            if eps >= 30:
                earnings_label = "💹 Earnings: Very Strong 💪"
            elif eps >= 15:
                earnings_label = "💹 Earnings: Strong ✅"
            elif eps >= 5:
                earnings_label = "💹 Earnings: Moderate ⚠️"
            else:
                earnings_label = "💹 Earnings: Weak ❌"

            # Valuation Label
            if pe_ratio < 10:
                valuation_label = "💰 Valuation: Undervalued 🟢"
            elif 10 <= pe_ratio <= 20:
                valuation_label = "💰 Valuation: Reasonable 👍"
            else:
                valuation_label = "💰 Valuation: Overvalued 🔴"

            # Risk Label
            if pbv < 1:
                risk_label = "📉 Risk Level: Low 🛡️"
            elif pbv < 2:
                risk_label = "📉 Risk Level: Moderate ⚠️"
            else:
                risk_label = "📉 Risk Level: High 🔥"
        except Exception as e:
            logger.error(f"Error generating labels: {e}")
            earnings_label = "💹 Earnings: N/A"
            valuation_label = "💰 Valuation: N/A"
            risk_label = "📉 Risk Level: N/A"

        return f"<br><br>🔍 Summary:<br>{earnings_label}<br>{valuation_label}<br>{risk_label}<br>"

    def format_fundamental_table(self, companies_data: List[Dict]) -> Tuple[str, str]:
        """Format a table for fundamental data with LLM insights and next steps."""
        thresholds = {
            "EPS": 20,
            "P/E Ratio": 15,
            "PBV": 1.5,
            "Market Capitalization": 10000000000
        }
        meanings = {
            "EPS": "Higher EPS = More Profitability",
            "P/E Ratio": "Low P/E = Potential undervaluation",
            "PBV": "Lower PBV = Stock may be undervalued",
            "Market Capitalization": "Larger cap = More stability"
        }

        company_names = [data.get("Company", f"Stock {i+1}") for i, data in enumerate(companies_data)]
        table = "<table border='1'><tr><th>Metric</th>" + "".join(
            f"<th>{name}</th>" for name in company_names
        ) + "<th>Meaning</th></tr>"

        metrics = ["EPS", "P/E Ratio", "PBV", "Market Capitalization"]
        metric_values = {metric: [data.get(metric, "N/A") for data in companies_data] for metric in metrics}

        for metric in metrics:
            row = f"<td>{metric}</td>"
            for value in metric_values[metric]:
                color = "black"
                if value != "N/A" and metric in thresholds:
                    try:
                        val = float(value)
                        if metric == "EPS" and val > thresholds[metric]:
                            color = "green"
                        elif metric == "EPS" and val < 10:
                            color = "red"
                        elif metric == "P/E Ratio" and val < thresholds[metric]:
                            color = "green"
                        elif metric == "P/E Ratio" and val > 20:
                            color = "red"
                        elif metric == "PBV" and val < thresholds[metric]:
                            color = "green"
                        elif metric == "PBV" and val > 2:
                            color = "red"
                        elif metric == "Market Capitalization" and val > thresholds[metric]:
                            color = "green"
                        elif metric == "Market Capitalization" and val < 1000000000:
                            color = "red"
                    except ValueError:
                        pass
                row += f"<td style='color:{color}'>{value}</td>"
            row += f"<td>{meanings[metric]}</td>"
            table += f"<tr>{row}</tr>"

        table += "</table>"

        prompt = "Based on this data, provide a fundamental analysis in simple language. Mention earnings strength, valuation (P/E, PBV), and overall financial health."
        formatted_data_text = "\n".join([f"{k}: {v}" for company in companies_data for k, v in company.items()])
        full_prompt = prompt + "\n" + formatted_data_text
        insights = self.get_llm_analysis(full_prompt)

        labels = self.generate_labels(companies_data[0])
        company_type = companies_data[0].get("Company Type", None)
        next_steps = self.generate_next_steps(company_names[0], company_type)

        return table, insights + labels + next_steps

    def format_fundamental_comparison(self, companies_data: List[Dict]) -> Tuple[str, str]:
        """Compare fundamental metrics of companies and provide insights."""
        thresholds = {
            "EPS": 20,
            "P/E Ratio": 15,
            "PBV": 1.5,
            "Market Capitalization": 10000000000
        }
        meanings = {
            "EPS": "Higher EPS = More Profitability",
            "P/E Ratio": "Low P/E = Potential undervaluation",
            "PBV": "Lower PBV = May be undervalued",
            "Market Capitalization": "Higher = Stability & investor confidence"
        }

        company_names = [data.get("Company", f"Stock {i+1}") for i, data in enumerate(companies_data)]
        metrics = ["EPS", "P/E Ratio", "PBV", "Market Capitalization"]
        metric_values = {metric: [data.get(metric, "N/A") for data in companies_data] for metric in metrics}

        table = "<table border='1'><tr><th>Metric</th>" + "".join(
            f"<th>{name}</th>" for name in company_names
        ) + "<th>Meaning</th></tr>"

        for metric in metrics:
            row = f"<td>{metric}</td>"
            for value in metric_values[metric]:
                color = "black"
                try:
                    val = float(value) if isinstance(value, (int, float, str)) and str(value).replace(".", "").isdigit() else None
                    if val is not None:
                        if metric == "EPS":
                            color = "green" if val > thresholds[metric] else "red"
                        elif metric == "P/E Ratio":
                            color = "green" if val < thresholds[metric] else "red" if val > 20 else "black"
                        elif metric == "PBV":
                            color = "green" if val < thresholds[metric] else "red" if val > 2 else "black"
                        elif metric == "Market Capitalization":
                            color = "green" if val > thresholds[metric] else "red" if val < 1e9 else "black"
                except:
                    pass
                row += f"<td style='color:{color}'>{value}</td>"
            row += f"<td>{meanings[metric]}</td>"
            table += f"<tr>{row}</tr>"

        table += "</table>"

        formatted_data_text = "\n".join([f"{k}: {v}" for company in companies_data for k, v in company.items()])
        full_prompt = (
            "Based on this data, provide a fundamental comparison between the companies in simple language. "
            "Mention earnings strength, valuation (P/E, PBV), and overall financial health.\n" + formatted_data_text
        )
        insight = self.get_llm_analysis(full_prompt)

        next_steps = (
            "<br><br>Would you like me to:<br>"
            "👉 Show their technical indicators (like RSI, MACD)?<br>"
            "👉 Compare them with a third company like NICA?<br>"
        )

        return table, insight + next_steps

    def get_daily_summary(self, stock_data: Dict, date: str, stock_name: str = "") -> str:
        """Generate a daily stock summary with price and technical indicators."""
        try:
            day = stock_data.get(date)
            if not day:
                return f"No data available for {date}."

            price = day["price"]
            indicators = day.get("indicators", {})

            summary = f"\nDate: {date}\n"
            if price['open'] > price['prevClose']:
                summary += f"Open: Rs. \033[32m{price['open']}\033[0m\n"
            else:
                summary += f"Open: Rs. \033[31m{price['open']}\033[0m\n"
            
            if price['close'] > price['open']:
                summary += f"Close: Rs. \033[32m{price['close']}\033[0m ({'⬆' if price['diff'] > 0 else '⬇'} Rs. {abs(price['diff'])} from previous close)\n"
            else:
                summary += f"Close: Rs. \033[31m{price['close']}\033[0m ({'⬆' if price['diff'] > 0 else '⬇'} Rs. {abs(price['diff'])} from previous close)\n"
            
            summary += f"High: Rs. {price['max']} | Low: Rs. {price['min']}\n"
            summary += f"Volume: {day['tradedShares']} shares | Amount: Rs. {day['amount']}\n"
            summary += f"Indicators:\n"
            summary += f"- SMA: {round(indicators.get('SMA', 0), 2)} | EMA: {round(indicators.get('EMA', 0), 2)}\n"

            rsi = indicators.get("RSI", None)
            if rsi is not None:
                rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                summary += f"- RSI: {round(rsi, 2)} ({rsi_status})\n"

            summary += (
                f"- MACD: {round(indicators.get('MACD', 0), 2)} | "
                f"Signal: {round(indicators.get('MACD_Signal', 0), 2)} | "
                f"Histogram: {round(indicators.get('MACD_Hist', 0), 2)}\n"
            )
            summary += (
                f"- BB High: {round(indicators.get('BB_High', 0), 2)} | "
                f"Mid: {round(indicators.get('BB_Mid', 0), 2)} | "
                f"Low: {round(indicators.get('BB_Low', 0), 2)}\n"
            )

            analysis_prompt = (
                f"Based on the technical indicators for {stock_name} on {date}, "
                "generate a short analysis explaining the stock's technical state, such as trend, momentum, or volatility."
            )
            ai_analysis = self.get_llm_analysis(analysis_prompt)

            next_steps = (
                "\n\nWould you like to:\n"
                "👉 View its technical indicator trends over the past week?\n"
                "👉 See its fundamental analysis?\n"
                "👉 Compare it with another stock?\n"
            )

            return summary + "\n\nAI Insight:\n" + ai_analysis.strip() + next_steps
        except Exception as e:
            logger.error(f"Error generating daily summary: {e}")
            return f"Error generating daily summary: {e}"

    def weekly_summary(self, weekly_data: List[Dict], stock_name: str) -> Tuple[str, str]:
        if not weekly_data or not isinstance(weekly_data, list):
            return f"No valid weekly data available for {stock_name}.", ""

        meanings = {
            "Close": "Green = Price went up compared to open",
            "RSI": "RSI > 70 = Overbought, < 30 = Oversold",
            "MACD": "Positive = Bullish momentum",
            "EMA": "Rising EMA = Positive trend",
            "Flags": "Key signals like MACD crossover or price action"
        }

        metrics = ["Open", "Close", "High", "Low", "RSI", "MACD", "EMA", "Flags"]
        table = f"<h3>Weekly Technical Trend Summary: {stock_name}</h3>"
        table += "<table border='1'><tr><th>Date</th>" + "".join(
            f"<th>{metric}</th>" for metric in metrics
        ) + "<th>Meaning</th></tr>"

        for day in weekly_data:
            row = f"<td>{day.get('date')}</td>"
            for metric in metrics:
                if metric == "Open":
                    value = day.get("price", {}).get("open", "N/A")
                elif metric == "Close":
                    value = day.get("price", {}).get("close", "N/A")
                elif metric == "High":
                    value = day.get("price", {}).get("max", "N/A")
                elif metric == "Low":
                    value = day.get("price", {}).get("min", "N/A")
                elif metric in ["RSI", "MACD", "EMA"]:
                    value = day.get("indicators", {}).get(metric, "N/A")
                else:
                    value = "N/A"  # Flags not present in data
                color = "black"
                try:
                    if metric == "Close" and isinstance(value, (int, float)):
                        color = "green" if value > float(day.get("price", {}).get("open", 0)) else "red"
                    elif metric == "RSI" and isinstance(value, (int, float)):
                        color = "red" if value > 70 else "green" if value < 30 else "black"
                    elif metric == "MACD" and isinstance(value, (int, float)):
                        color = "green" if value > 0 else "red"
                    elif metric == "EMA" and isinstance(value, (int, float)):
                        color = "green" if value > float(day.get("price", {}).get("open", 0)) else "red"
                except:
                    pass
                row += f"<td style='color:{color}'>{value}</td>"
            row += f"<td>{meanings.get(metric, '')}</td>"
            table += f"<tr>{row}</tr>"

        table += "</table>"

        prompt = f"Analyze this weekly trend data for {stock_name} and provide insights in simple language. Highlight trend strength, momentum signals (MACD, RSI), and general market behavior."
        formatted_data = "\n".join([
            f"{d['date']} - Close: {d.get('price', {}).get('close', 'N/A')}, "
            f"RSI: {d.get('indicators', {}).get('RSI', 'N/A')}, "
            f"MACD: {d.get('indicators', {}).get('MACD', 'N/A')}, "
            f"EMA: {d.get('indicators', {}).get('EMA', 'N/A')}, "
            f"Flags: {'N/A'}"
            for d in weekly_data
        ])
        full_prompt = prompt + "\n" + formatted_data
        insights = self.get_llm_analysis(full_prompt)

        return table, insights

    def get_volatility_analysis(self, stock_data: Dict, end_date: str, stock_name: str = "") -> Tuple[str, str]:
        if not stock_data or "data" not in stock_data or not isinstance(stock_data["data"], list):
            return f"No valid stock data available for {stock_name}.", ""
        daily_data = sorted(stock_data.get("data", []), key=lambda x: x["date"])
        recent_data = [d for d in daily_data if d["date"] <= end_date][-14:]

        if len(recent_data) < 5:
            return f"Not enough data to perform volatility analysis for {stock_name}.", ""

        returns = []
        true_ranges = []
        for i in range(1, len(recent_data)):
            prev = recent_data[i - 1]
            curr = recent_data[i]
            try:
                close_prev = float(prev["price"]["close"])
                close_curr = float(curr["price"]["close"])
                returns.append((close_curr - close_prev) / close_prev * 100)
                high = float(curr["price"]["max"])
                low = float(curr["price"]["min"])
                true_ranges.append(high - low)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Skipping invalid data for {stock_name} on {curr.get('date', 'unknown')}: {e}")
                continue

        if not returns or not true_ranges:
            return f"Insufficient valid data for volatility analysis of {stock_name}.", ""

        avg_return = np.mean(returns)
        std_dev = np.std(returns)
        atr = np.mean(true_ranges)

        if std_dev < 1:
            vol_class = "Low"
            color = "green"
        elif std_dev < 2.5:
            vol_class = "Medium"
            color = "orange"
        else:
            vol_class = "High"
            color = "red"

        report = f"""
        <h3>Volatility Analysis for {stock_name}</h3>
        <table border='1' style='border-collapse:collapse;'>
            <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
            <tr><td>Average Daily Return</td><td>{avg_return:.2f}%</td><td>{'Positive return trend' if avg_return > 0 else 'Negative return trend'}</td></tr>
            <tr><td>Standard Deviation of Returns</td><td>{std_dev:.2f}%</td><td>Higher = More volatile, Lower = More stable</td></tr>
            <tr><td>Average True Range (ATR)</td><td>{atr:.2f}</td><td>Daily price fluctuation magnitude</td></tr>
            <tr><td>Volatility Classification</td><td style='color:{color}'><b>{vol_class}</b></td><td>Overall price movement behavior</td></tr>
        </table>
        """

        prompt = (
            f"Analyze the following volatility metrics for {stock_name}:\n"
            f"- Average Return: {avg_return:.2f}%\n"
            f"- Standard Deviation: {std_dev:.2f}%\n"
            f"- ATR: {atr:.2f}\n"
            f"- Volatility: {vol_class}\n"
            "Provide a brief summary on what this means for an investor or trader."
        )
        insights = self.get_llm_analysis(prompt)

        return report, insights


    def get_technical_indicator_summary(self, stock_data: Dict, stock_name: str = "") -> Tuple[str, str]:
        """Generate a summary of technical indicators (RSI, MACD, SMA, EMA, Bollinger Bands)."""
        daily_data = sorted(stock_data.get("data", []), key=lambda x: x["date"])
        if len(daily_data) < 5:
            return f"Not enough data to perform technical indicator analysis for {stock_name}.", ""

        rsi = daily_data[-1].get("indicators", {}).get("RSI", None)
        macd = daily_data[-1].get("indicators", {}).get("MACD", None)
        sma = daily_data[-1].get("indicators", {}).get("SMA", None)
        ema = daily_data[-1].get("indicators", {}).get("EMA", None)
        bb_high = daily_data[-1].get("indicators", {}).get("BB_High", None)
        bb_low = daily_data[-1].get("indicators", {}).get("BB_Low", None)
        bb_mid = daily_data[-1].get("indicators", {}).get("BB_Mid", None)

        table = f"""
        <h3>Technical Indicator Summary for {stock_name}</h3>
        <table border='1' style='border-collapse:collapse;'>
            <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
        """

        if rsi is not None:
            rsi_interpretation = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            table += f"<tr><td>RSI</td><td>{rsi:.2f}</td><td>{rsi_interpretation}</td></tr>"
        else:
            table += "<tr><td>RSI</td><td>Not enough data</td><td>RSI requires at least 14 days of data to calculate.</td></tr>"

        if macd is not None:
            macd_interpretation = "Bullish" if macd > 0 else "Bearish" if macd < 0 else "Neutral"
            table += f"<tr><td>MACD</td><td>{macd:.2f}</td><td>{macd_interpretation}</td></tr>"
        else:
            table += "<tr><td>MACD</td><td>Not enough data</td><td>MACD requires at least 26 days of data to calculate.</td></tr>"

        if sma is not None:
            sma_interpretation = "Uptrend" if sma > daily_data[-2]["price"]["close"] else "Downtrend"
            table += f"<tr><td>SMA</td><td>{sma:.2f}</td><td>{sma_interpretation}</td></tr>"
        else:
            table += "<tr><td>SMA</td><td>Not enough data</td><td>SMA requires at least 20 days of data to calculate.</td></tr>"

        if ema is not None:
            ema_interpretation = "Uptrend" if ema > daily_data[-2]["price"]["close"] else "Downtrend"
            table += f"<tr><td>EMA</td><td>{ema:.2f}</td><td>{ema_interpretation}</td></tr>"
        else:
            table += "<tr><td>EMA</td><td>Not enough data</td><td>EMA requires at least 26 days of data to calculate.</td></tr>"

        if bb_high is not None and bb_low is not None:
            bb_interpretation = "Overbought" if daily_data[-1]["price"]["close"] > bb_high else "Oversold" if daily_data[-1]["price"]["close"] < bb_low else "Normal"
            table += f"<tr><td>Bollinger Bands</td><td>High: {bb_high:.2f}, Low: {bb_low:.2f}</td><td>{bb_interpretation}</td></tr>"
        else:
            table += "<tr><td>Bollinger Bands</td><td>Not enough data</td><td>Bollinger Bands require at least 20 days of data to calculate.</td></tr>"

        table += "</table>"

        prompt = (
            f"Analyze the following technical indicators for {stock_name}:\n"
            f"- RSI: {rsi if rsi else 'Not available'}\n"
            f"- MACD: {macd if macd else 'Not available'}\n"
            f"- SMA: {sma if sma else 'Not available'}\n"
            f"- EMA: {ema if ema else 'Not available'}\n"
            f"- Bollinger Bands (High: {bb_high if bb_high else 'Not available'}, Low: {bb_low if bb_low else 'Not available'})\n"
            "Provide a brief summary of the stock's technical outlook."
        )
        insights = self.get_llm_analysis(prompt)

        return table, insights

    def get_stock_comparison(self, stock_data_1: Dict, stock_data_2: Dict, stock_name_1: str, stock_name_2: str, date: Optional[str] = None) -> Tuple[str, str]:
        """Compare stock indicators for two companies."""
        if not stock_data_1 or "data" not in stock_data_1 or not stock_data_1["data"]:
            return f"No data available for {stock_name_1}.", ""
        if not stock_data_2 or "data" not in stock_data_2 or not stock_data_2["data"]:
            return f"No data available for {stock_name_2}.", ""

        if date:
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                return f"Invalid date format: {date}. Please use YYYY-MM-DD.", ""

        data_1 = {entry['date']: entry for entry in stock_data_1.get('data', [])}
        data_2 = {entry['date']: entry for entry in stock_data_2.get('data', [])}

        if not date:
            common_dates = sorted(set(data_1.keys()) & set(data_2.keys()))
            if not common_dates:
                return f"No overlapping date records found between {stock_name_1} and {stock_name_2}.", ""
            date = common_dates[-1]

        entry_1 = data_1.get(date)
        entry_2 = data_2.get(date)

        if not entry_1 or not entry_2:
            missing = []
            if not entry_1:
                missing.append(stock_name_1)
            if not entry_2:
                missing.append(stock_name_2)
            return f"Data for {', '.join(missing)} is not available on {date}.", ""

        def safe_get(val, precision=2):
            return f"{val:.{precision}f}" if isinstance(val, (int, float)) else "N/A"

        p1, i1 = entry_1.get("price", {}), entry_1.get("indicators", {})
        p2, i2 = entry_2.get("price", {}), entry_2.get("indicators", {})

        table = f"""
        <h3>Stock Comparison on {date}</h3>
        <table border='1' style='border-collapse:collapse;'>
            <tr><th>Metric</th><th>{stock_name_1}</th><th>{stock_name_2}</th></tr>
            <tr><td>Closing Price</td><td>{safe_get(p1.get('close'))}</td><td>{safe_get(p2.get('close'))}</td></tr>
            <tr><td>Price Change</td><td>{safe_get(p1.get('diff'))}</td><td>{safe_get(p2.get('diff'))}</td></tr>
            <tr><td>RSI</td><td>{safe_get(i1.get('RSI'))}</td><td>{safe_get(i2.get('RSI'))}</td></tr>
            <tr><td>MACD</td><td>{safe_get(i1.get('MACD'))}</td><td>{safe_get(i2.get('MACD'))}</td></tr>
            <tr><td>SMA</td><td>{safe_get(i1.get('SMA'))}</td><td>{safe_get(i2.get('SMA'))}</td></tr>
            <tr><td>EMA</td><td>{safe_get(i1.get('EMA'))}</td><td>{safe_get(i2.get('EMA'))}</td></tr>
            <tr><td>Volatility (BB Width)</td>
                <td>{safe_get(i1.get('BB_High') - i1.get('BB_Low')) if i1.get('BB_High') and i1.get('BB_Low') else 'N/A'}</td>
                <td>{safe_get(i2.get('BB_High') - i2.get('BB_Low')) if i2.get('BB_High') and i2.get('BB_Low') else 'N/A'}</td>
            </tr>
        </table>
        """

        summary = (
            f"On {date}, {stock_name_1} closed at NPR {safe_get(p1.get('close'))} "
            f"with RSI {safe_get(i1.get('RSI'))}, and {stock_name_2} closed at NPR {safe_get(p2.get('close'))} "
            f"with RSI {safe_get(i2.get('RSI'))}. This suggests comparative momentum and technical positioning. "
            f"Always consider long-term context and sector factors when making decisions."
        )

        return table, summary

    def get_monthly_performance_summary(self, stock_data: Dict, stock_name: str, month: str) -> Tuple[str, str]:
        """Generate a monthly performance summary for a stock."""
        if not stock_data or "data" not in stock_data or not stock_data["data"]:
            return f"No data available for {stock_name}.", ""

        try:
            datetime.strptime(month, "%Y-%m")
        except ValueError:
            return f"Invalid month format: {month}. Please use YYYY-MM format.", ""

        monthly_entries = [entry for entry in stock_data["data"] if entry["date"].startswith(month)]
        if not monthly_entries:
            return f"No records found for {stock_name} in {month}.", ""

        monthly_entries.sort(key=lambda x: x["date"])
        first_day = monthly_entries[0]
        last_day = monthly_entries[-1]

        def safe_avg(entries, key):
            values = [e.get("indicators", {}).get(key) for e in entries if isinstance(e.get("indicators", {}).get(key), (int, float))]
            return sum(values) / len(values) if values else None

        def bb_width(entry):
            bb_high = entry.get("indicators", {}).get("BB_High")
            bb_low = entry.get("indicators", {}).get("BB_Low")
            return bb_high - bb_low if isinstance(bb_high, (int, float)) and isinstance(bb_low, (int, float)) else None

        open_price = first_day.get("price", {}).get("close")
        close_price = last_day.get("price", {}).get("close")
        price_change = close_price - open_price if open_price and close_price else None

        avg_close_price = sum(
            e.get("price", {}).get("close", 0) for e in monthly_entries if isinstance(e.get("price", {}).get("close"), (int, float))
        ) / len(monthly_entries)

        avg_rsi = safe_avg(monthly_entries, "RSI")
        avg_sma = safe_avg(monthly_entries, "SMA")
        avg_ema = safe_avg(monthly_entries, "EMA")
        avg_volatility = sum(
            bb_width(e) for e in monthly_entries if bb_width(e) is not None
        ) / len([e for e in monthly_entries if bb_width(e) is not None]) if any(bb_width(e) for e in monthly_entries) else None

        def safe_fmt(val, precision=2):
            return f"{val:.{precision}f}" if isinstance(val, (int, float)) else "N/A"

        table = f"""
        <h3>{stock_name} Monthly Performance Summary - {month}</h3>
        <table border='1' style='border-collapse: collapse;'>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Opening Price</td><td>{safe_fmt(open_price)}</td></tr>
            <tr><td>Closing Price</td><td>{safe_fmt(close_price)}</td></tr>
            <tr><td>Price Change</td><td>{safe_fmt(price_change)}</td></tr>
            <tr><td>Average Close Price</td><td>{safe_fmt(avg_close_price)}</td></tr>
            <tr><td>Average RSI</td><td>{safe_fmt(avg_rsi)}</td></tr>
            <tr><td>Average SMA</td><td>{safe_fmt(avg_sma)}</td></tr>
            <tr><td>Average EMA</td><td>{safe_fmt(avg_ema)}</td></tr>
            <tr><td>Average Volatility (BB Width)</td><td>{safe_fmt(avg_volatility)}</td></tr>
        </table>
        """

        summary = (
            f"In {month}, {stock_name} showed a net price change of NPR {safe_fmt(price_change)} "
            f"from NPR {safe_fmt(open_price)} to NPR {safe_fmt(close_price)}. "
            f"Technical indicators like RSI (avg {safe_fmt(avg_rsi)}) and EMA (avg {safe_fmt(avg_ema)}) "
            f"provide a trend snapshot. Volatility averaged around {safe_fmt(avg_volatility)}. "
            f"This helps assess risk and momentum for the month."
        )

        return table, summary

    def get_trend_analysis(self, stock_data: Dict, stock_name: str = "", days: int = 7) -> Tuple[str, str]:
        if not stock_data or "data" not in stock_data or not isinstance(stock_data["data"], list):
            return f"No valid stock data available for {stock_name}.", ""
        entries = stock_data["data"][-days:]
        if len(entries) < days:
            return f"Insufficient recent data to analyze trend for {stock_name} (need {days} days, got {len(entries)}).", ""
        try:
            closes = [e["price"]["close"] for e in entries if "price" in e and "close" in e["price"]]
            smas = [e["indicators"].get("SMA") for e in entries if "indicators" in e]
            emas = [e["indicators"].get("EMA") for e in entries if "indicators" in e]
            rsis = [e["indicators"].get("RSI") for e in entries if "indicators" in e]
            macds = [e["indicators"].get("MACD") for e in entries if "indicators" in e]
            signals = [e["indicators"].get("MACD_Signal") for e in entries if "indicators" in e]

            if not closes or not rsis or not smas or not emas:
                return f"Missing required data (prices or indicators) for {stock_name}.", ""

            avg_rsi = mean([r for r in rsis if r is not None])
            avg_macd_diff = mean([(m - s) for m, s in zip(macds, signals) if m is not None and s is not None])
            price_trend = closes[-1] - closes[0]
            avg_ema = mean([e for e in emas if e is not None])
            avg_sma = mean([s for s in smas if s is not None])

            if price_trend > 0 and avg_rsi > 55 and avg_ema > avg_sma and avg_macd_diff > 0:
                trend = "Uptrend 📈"
            elif price_trend < 0 and avg_rsi < 45 and avg_ema < avg_sma and avg_macd_diff < 0:
                trend = "Downtrend 📉"
            else:
                trend = "Sideways 🔄"

            html = f"""
            <h3>{stock_name} Trend Analysis - Last {days} Days</h3>
            <table border='1' style='border-collapse:collapse;'>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Trend</td><td>{trend}</td></tr>
                <tr><td>Price Change</td><td>{price_trend:.2f} NPR</td></tr>
                <tr><td>Avg RSI</td><td>{avg_rsi:.2f}</td></tr>
                <tr><td>Avg EMA</td><td>{avg_ema:.2f}</td></tr>
                <tr><td>Avg SMA</td><td>{avg_sma:.2f}</td></tr>
                <tr><td>Avg MACD - Signal</td><td>{avg_macd_diff:.2f}</td></tr>
            </table>
            """

            description = (
                f"Over the past {days} days, {stock_name} shows a **{trend.lower()}** pattern, "
                f"with prices changing by {price_trend:.2f} NPR. The average RSI was {avg_rsi:.2f}, "
                f"suggesting {'bullish' if avg_rsi > 55 else 'bearish' if avg_rsi < 45 else 'neutral'} momentum. "
                f"Additionally, EMA vs SMA and MACD vs Signal indicate overall technical {trend.lower()}."
            )

            return html, description
        except Exception as e:
            logger.error(f"Error analyzing trend for {stock_name}: {e}")
            return f"Error analyzing trend for {stock_name}: {str(e)}", ""
        

    def route_prompt(self, user_input: str) -> Dict:
        # 1. Language check: Only accept English or Nepali
        lang = self.detect_language(user_input)
        if lang not in ["en", "ne"]:
            return {"function": None, "params": {}, "message": "Only English and Nepali queries are supported."}

        # 2. Translate if Nepali
        translated = self.perform_translation(user_input, "ne", "en") if lang == "ne" else user_input

        # 3. Extract + fuzzy match tickers
        extracted_words = re.findall(r"\b[A-Za-z]{2,15}\b", translated)
        tickers = []
        for word in extracted_words:
            if word.upper() in self.VALID_TICKERS:
                tickers.append(word.upper())
            else:
                match = self.fuzzy_match_ticker(word)
                if match:
                    tickers.append(match)
        tickers = list(set(tickers))  # Remove duplicates

        # 4. Extract date & month
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", translated)
        date = date_match.group(0) if date_match else None
        month_match = re.search(r"\d{4}-\d{2}", translated)
        month = month_match.group(0) if month_match else None

        logger.info(f"Processed input: '{translated}' | Valid tickers: {tickers}")

        # 5. Keyword-Based Routing
        input_lower = translated.lower()
        for rule in self.routing_rules:
            if all(kw in input_lower for kw in rule["keywords"]) and len(tickers) >= rule["min_tickers"]:
                params = {"symbols": tickers[:2]} if "symbols" in rule["function"] else {"stock_name": tickers[0]}
                if "date" in rule["function"]:
                    params["date"] = date
                if "month" in rule["function"]:
                    params["month"] = month
                if "end_date" in rule["function"]:
                    params["end_date"] = date
                return {"function": rule["function"], "params": params}

        # 6. Semantic fallback: classify with LLM intent
        intent = self.classify_intent_with_llm(translated)
        fallback_map = {
            "trend": "get_trend_analysis",
            "comparison": "get_stock_comparison",
            "volatility": "get_volatility_analysis",
            "fundamentals": "format_fundamental_table",
            "technical": "get_technical_indicator_summary",
            "daily": "get_daily_summary"
        }
        if intent in fallback_map and tickers:
            func = fallback_map[intent]
            return {"function": func, "params": {"stock_name": tickers[0], "date": date, "month": month, "end_date": date}}

        return {"function": None, "params": {}, "message": "Unrecognized query."}

    # Helper for error messages
    def error(self, message: str) -> Dict:
        return {"function": None, "params": {}, "message": f"{message}"}



    def load_fundamental_data(self) -> List[Dict]:
        """Load fundamental data from CSV file."""
        try:
            base_path = Path(__file__).parent.parent
            csv_path = base_path / "data" / "initial data" / "fundamental_data.csv"
            df = pd.read_csv(csv_path)
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error loading fundamental data: {e}")
            return []
        
    def load_stock_data(self, symbol: str) -> Dict:
        """Load stock price/indicator data from JSON files."""
        try:
            base_path = Path(__file__).parent.parent
            data_dir = base_path / "data" / "stock_data"
            file_path = data_dir / f"{symbol.upper()}.json"
            
            if not file_path.exists():
                logger.warning(f"No stock data found for {symbol}")
                return {}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading stock data for {symbol}: {e}")
            return {}
        
    def get_llm_analysis(self, prompt: str) -> str:
        """Generate insights using the TinyLlama model."""
        try:
            formatted_prompt = f"""<|system|>
You are a financial assistant. Summarize this input into 3–5 lines with a clear explanation. Highlight momentum, volatility, trend, or valuation. Avoid repeating the input.</s>
<|user|>
{prompt}</s>
<|assistant|>"""

            inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.device)
            generation_kwargs = {
                **inputs,
                "streamer": self.streamer,
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": 0.95,
                "top_k": 40,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.pad_token_id
            }

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            response = ""
            timeout = 30
            start = time.time()
            while thread.is_alive() and (time.time() - start) < timeout:
                for token in self.streamer:
                    if token.strip() not in ["</s>", "<|system|>", "<|user|>", "<|assistant|>"]:
                        response += token
                time.sleep(0.1)

            if thread.is_alive():
                thread.join()
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating LLM insight: {e}")
            return "⚠️ AI Insight could not be generated."

    def make_predictions(self, question: str) -> str:
        try:
            route = self.route_prompt(question)
            func_name = route["function"]
            params = route["params"]

            if "stock_name" in params:
                params["stock_data"] = self.load_stock_data(params["stock_name"])
            if "stock_name_1" in params:
                params["stock_data_1"] = self.load_stock_data(params["stock_name_1"])
            if "stock_name_2" in params:
                params["stock_data_2"] = self.load_stock_data(params["stock_name_2"])

            if func_name == "get_daily_summary":
                return self.get_daily_summary(**params)
            elif func_name == "weekly_summary":
                weekly_data = params["stock_data"].get("data", [])[-7:]  # Last 7 days
                if not weekly_data:
                    return f"No weekly data available for {params['stock_name']}.", ""
                table, insights = self.weekly_summary(weekly_data, params["stock_name"])
                return table + "<br><br>" + insights
            elif func_name == "get_volatility_analysis":
                table, insights = self.get_volatility_analysis(**params)
                return table + "<br><br>" + insights
            elif func_name == "get_monthly_performance_summary":
                table, summary = self.get_monthly_performance_summary(**params)
                return table + "<br><br>" + summary
            elif func_name == "get_technical_indicator_summary":
                table, insights = self.get_technical_indicator_summary(**params)
                return table + "<br><br>" + insights
            elif func_name == "get_trend_analysis":
                table, description = self.get_trend_analysis(**params)
                return table + "<br><br>" + description
            elif func_name == "get_stock_comparison":
                table, summary = self.get_stock_comparison(**params)
                return table + "<br><br>" + summary
            elif func_name == "format_fundamental_comparison":
                symbols = params.get("symbols", [])
                fundamentals = self.load_fundamental_data()
                filtered = [f for f in fundamentals if f.get("Company Symbol") in symbols]
                if len(filtered) < 2:
                    return "❌ One or both companies not found in the fundamental data."
                table, summary = self.format_fundamental_comparison(filtered)
                return table + "<br><br>" + summary
            elif func_name == "format_fundamental_table":
                symbols = params.get("symbols", [])
                fundamentals = self.load_fundamental_data()
                filtered = [f for f in fundamentals if f.get("Company Symbol") in symbols]
                if not filtered:
                    return "❌ Company not found in the fundamental data."
                table, summary = self.format_fundamental_table(filtered)
                return table + "<br><br>" + summary
            elif func_name is None:
                return route["message"]

            return self.get_llm_analysis(question)
        except Exception as e:
            logger.error(f"Error in make_predictions: {e}")
            return f"⚠️ An error occurred: {e}"
