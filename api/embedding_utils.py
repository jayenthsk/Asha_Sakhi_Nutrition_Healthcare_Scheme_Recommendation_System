import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from api.logging_utils import setup_logger

# Set up logger
logger = setup_logger(__name__)

class EmbeddingModel:
    def __init__(self):
        try:
            logger.info("Initializing embedding model")
            self.model_dir = os.path.join("models", "all-MiniLM-L6-v2")
            
            if not os.path.exists(self.model_dir):
                logger.warning(f"Model directory {self.model_dir} does not exist. Please run download_model.py first.")
                raise FileNotFoundError(f"Model directory {self.model_dir} not found")
            
            logger.info(f"Loading tokenizer from {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            logger.info(f"Loading model from {self.model_dir}")
            self.model = AutoModel.from_pretrained(self.model_dir)
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            self.model.to(self.device)
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling - take attention mask into account for correct averaging
        """
        try:
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        except Exception as e:
            logger.error(f"Error in mean pooling: {str(e)}")
            raise
    
    def encode(self, sentences, batch_size=32):
        """
        Generate embeddings for a list of sentences
        """
        if not sentences:
            logger.warning("Empty list of sentences provided for encoding")
            return []
        
        logger.info(f"Encoding {len(sentences)} sentences with batch size {batch_size}")
        all_embeddings = []
        
        try:
            # Process sentences in batches
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}/{(len(sentences)-1)//batch_size + 1}")
                
                # Tokenize sentences
                encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                
                # Compute token embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                
                # Perform pooling
                sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Normalize embeddings
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                
                # Convert to list and add to results
                all_embeddings.extend(sentence_embeddings.cpu().numpy().tolist())
            
            logger.info(f"Successfully encoded {len(sentences)} sentences")
            return all_embeddings
        except Exception as e:
            logger.error(f"Error encoding sentences: {str(e)}")
            raise

# Singleton instance
embedding_model = None

def get_embedding_model():
    """
    Get or create the embedding model instance
    """
    global embedding_model
    try:
        if embedding_model is None:
            logger.info("Creating new embedding model instance")
            embedding_model = EmbeddingModel()
        return embedding_model
    except Exception as e:
        logger.error(f"Error getting embedding model: {str(e)}")
        raise 