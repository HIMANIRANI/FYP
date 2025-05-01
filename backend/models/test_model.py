import logging
from pathlib import Path

from model import PredictionPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_data_files():
    """Verify that required data files exist."""
    base_path = Path(__file__).parent.parent
    required_files = [
        base_path / "data" / "updated_company_data" / "nabil.json",
        base_path / "data" / "updated_company_data" / "nica.json",
        base_path / "data" / "updated_company_data" / "akpl.json",
        base_path / "data" / "initial data" / "fundamental_data.csv"
    ]
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        logger.error(f"Missing required data files: {missing_files}")
        return False
    return True

def test_pipeline():
    """Test PredictionPipeline with provided queries."""
    try:
        # Verify data files
        if not verify_data_files():
            print("Error: Required data files are missing. Please check the logs.")
            return

        # Initialize pipeline
        logger.info("Initializing PredictionPipeline...")
        pipeline = PredictionPipeline()

        # Load components
        logger.info("Loading model and tokenizers...")
        pipeline.load_model_and_tokenizers()

        logger.info("Loading sentence transformer...")
        pipeline.load_sentence_transformer()

        logger.info("Loading reranking model...")
        pipeline.load_reranking_model()

        logger.info("Loading embeddings...")
        pipeline.load_embeddings()

        # Test query 1: Fundamental comparison
        logger.info("Testing: Compare fundamentals of NABIL and NICA")
        result = pipeline.make_predictions("Compare fundamentals of NABIL and NICA")
        print("Fundamental Comparison Result (NABIL vs NICA):")
        print(result)
        print("-" * 80)

        # Test query 2: Trend analysis for AKPL
        logger.info("Testing: What’s the trend of AKPL this week?")
        result = pipeline.make_predictions("What’s the trend of AKPL this week?")
        print("Trend Analysis Result for AKPL:")
        print(result)

        # # Test query 3: Fundamental metrics for NABIL
        # logger.info("Testing: What are the key fundamental metrics of NWCL for the last quarter?")
        # result = pipeline.make_predictions("What are the key fundamental metrics of NWCL for the last quarter?")
        # print("Fundamental Metrics Result (NWCL):")
        # print(result)
        # print("-" * 80)

        # logger.info("Testing: What are the key fundamental details of NWCL ?")
        # result = pipeline.make_predictions("What are the key fundamental details of NWCL ?")
        # print("Fundamental Metrics Result (NWCL):")
        # print(result)
        # print("-" * 80) 

        # # Test query 4: Volatility comparison between NICA and AKPL
        # logger.info("Testing: How volatile has NICA been compared to AKPL over the past month?")
        # result = pipeline.make_predictions("How volatile has NICA been compared to AKPL over the past month?")
        # print("Volatility Comparison Result (NICA vs AKPL):")
        # print(result)
        # print("-" * 80)

        # # Test query 5: Price prediction for NICA
        # logger.info("Testing: What is the predicted price movement for NICA in the next week?")
        # result = pipeline.make_predictions("What is the predicted price movement for NICA in the next week?")
        # print("Price Prediction Result (NICA):")
        # print(result)
        # print("-" * 80)

        # # Test query 6: Long-term trend for AKPL
        # logger.info("Testing: What has been the yearly performance trend of AKPL?")
        # result = pipeline.make_predictions("What has been the yearly performance trend of AKPL?")
        # print("Yearly Trend Analysis Result (AKPL):")
        # print(result)
        # print("-" * 80)

        # # Test query 7: Dividend yield comparison
        # logger.info("Testing: What is the dividend yield of NABIL compared to NICA?")
        # result = pipeline.make_predictions("What is the dividend yield of NABIL compared to NICA?")
        # print("Dividend Yield Comparison Result (NABIL vs NICA):")
        # print(result)
        # print("-" * 80)

        # # Test query 8: Price-to-earnings ratio comparison
        # logger.info("Testing: What is the price-to-earnings ratio of NABIL compared to AKPL?")
        # result = pipeline.make_predictions("What is the price-to-earnings ratio of NABIL compared to AKPL?")
        # print("Price-to-Earnings Ratio Comparison Result (NABIL vs AKPL):")
        # print(result)
        # print("-" * 80)
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_pipeline()
