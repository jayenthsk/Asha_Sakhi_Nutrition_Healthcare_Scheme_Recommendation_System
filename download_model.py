import os
import torch
from transformers import AutoTokenizer, AutoModel
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("download_model.log")
    ]
)

logger = logging.getLogger(__name__)

def download_model():
    """
    Download and save the sentence transformer model locally
    """
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        local_model_dir = os.path.join("models", "all-MiniLM-L6-v2")
        
        # Create the models directory if it doesn't exist
        os.makedirs(local_model_dir, exist_ok=True)
        logger.info(f"Created model directory: {local_model_dir}")
        
        logger.info(f"Downloading model {model_name}...")
        
        try:
            # Download and save the tokenizer
            logger.info("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_model_dir)
            tokenizer.save_pretrained(local_model_dir)
            logger.info("Tokenizer downloaded and saved successfully")
            
            # Download and save the model
            logger.info("Downloading model...")
            model = AutoModel.from_pretrained(model_name, cache_dir=local_model_dir)
            model.save_pretrained(local_model_dir)
            logger.info("Model downloaded and saved successfully")
            
            logger.info(f"Model saved to {local_model_dir}")
            return True
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Unexpected error in download_model: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        logger.info("Starting model download process")
        success = download_model()
        if success:
            logger.info("Model download completed successfully")
        else:
            logger.error("Model download failed")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}") 