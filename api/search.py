import os
import re
from qdrant_client import QdrantClient
from api.embedding_utils import get_embedding_model
from api.logging_utils import setup_logger

logger = setup_logger(__name__)

async def semantic_search(query: str, collection_name: str, limit: int = 3):
    """
    Perform semantic search on stored PDF content
    """
    logger.info(f"Performing semantic search with query: '{query}', collection: {collection_name}, limit: {limit}")
    
    try:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            logger.error("Qdrant URL or API key not found in environment variables")
            raise ValueError("Qdrant URL or API key not found in environment variables")
        
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        collections = client.get_collections().collections
        collection_exists = any(collection.name == collection_name for collection in collections)
        
        if not collection_exists:
            logger.error(f"Collection {collection_name} does not exist")
            raise ValueError(f"Collection {collection_name} does not exist")
        
        model = get_embedding_model()
        query_vector = model.encode([query])[0]
        
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        logger.info(f"Found {len(results)} results")
        
        formatted_results = []
        for i, res in enumerate(results):
            content = res.payload.get("text", "")
            structured_data = parse_health_scheme(content)
            structured_data["result_id"] = i + 1
            formatted_results.append(structured_data)
        
        logger.info("Search completed successfully")
        return {
            "results": formatted_results
        }
    
    except Exception as e:
        logger.error(f"Error in semantic_search: {str(e)}")
        raise

def parse_health_scheme(content):
    """
    Parse health scheme content into structured key-value pairs
    """
    structured_data = {
        "scheme_name": "",
        "state": "",
        "description": "",
        "eligibility": "",
        "how_to_apply": "",
        "benefits": "",
        "documents_required": "",
        "contact_info": ""
    }
    
    lines = content.strip().split('\n')
    if lines:
        structured_data["scheme_name"] = lines[0].strip()
    
    patterns = {
        "state": r"State:\s*([^\n]+)",
        "description": r"Description:\s*([^\n]+)",
        "eligibility": r"Eligibility:\s*([^\n]+)",
        "how_to_apply": r"How to Apply:\s*([^\n]+)",
        "benefits": r"Benefits:\s*([^\n]+)",
        "documents_required": r"Documents Required:\s*([^\n]+)",
        "contact_info": r"Contact:\s*([^\n]+)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            structured_data[key] = match.group(1).strip()
    
    if all(value == "" for key, value in structured_data.items() if key != "scheme_name" and key != "result_id"):
        structured_data["raw_content"] = content
    
    cleaned_data = {k: v for k, v in structured_data.items() if v != ""}
    
    return cleaned_data 