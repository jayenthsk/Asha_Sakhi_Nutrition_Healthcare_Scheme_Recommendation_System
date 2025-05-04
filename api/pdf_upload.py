import os
import uuid
import tempfile
from fastapi import UploadFile
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from api.embedding_utils import get_embedding_model
from api.logging_utils import setup_logger

# Set up logger
logger = setup_logger(__name__)

async def process_pdf(file: UploadFile, collection_name: str):
    """
    Process a PDF file, extract text, generate embeddings, and store in Qdrant
    """
    logger.info(f"Processing PDF file: {file.filename}, collection: {collection_name}")
    
    try:
        # Load environment variables
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            logger.error("Qdrant URL or API key not found in environment variables")
            raise ValueError("Qdrant URL or API key not found in environment variables")
        
        chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
        
        logger.info(f"Using chunk size: {chunk_size}, chunk overlap: {chunk_overlap}")
        
        # Initialize Qdrant client
        logger.info(f"Connecting to Qdrant at {qdrant_url}")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Check if collection exists, if not create it
        logger.info(f"Checking if collection {collection_name} exists")
        collections = client.get_collections().collections
        collection_exists = any(collection.name == collection_name for collection in collections)
        
        if not collection_exists:
            logger.info(f"Creating new collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        
        # Get the embedding model
        logger.info("Loading embedding model")
        model = get_embedding_model()
        
        # Save the uploaded file to a temporary location
        logger.info("Saving uploaded file to temporary location")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            logger.info(f"Temporary file saved at: {temp_file_path}")
        
        try:
            # Load and process the PDF
            logger.info(f"Loading PDF from {temp_file_path}")
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load()
            
            if not pages:
                logger.warning(f"No content found in {file.filename}")
                return {"status": "warning", "message": f"No content found in {file.filename}."}
            
            logger.info(f"Loaded {len(pages)} pages from PDF")
            
            # Split text into chunks
            logger.info("Splitting text into chunks")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            chunks = []
            for page in pages:
                page_chunks = text_splitter.split_text(page.page_content)
                for i, chunk in enumerate(page_chunks):
                    chunks.append({
                        "text": chunk,
                        "source": file.filename,
                        "page": page.metadata.get("page", 0) + 1,
                        "chunk": i + 1
                    })
            
            if not chunks:
                logger.warning(f"No text chunks extracted from {file.filename}")
                return {"status": "warning", "message": f"No text chunks extracted from {file.filename}."}
            
            logger.info(f"Created {len(chunks)} text chunks")
            
            # Generate embeddings for each chunk
            logger.info("Generating embeddings for chunks")
            texts = [chunk["text"] for chunk in chunks]
            embeddings = model.encode(texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Create points for Qdrant
            logger.info("Creating points for Qdrant")
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[i],
                    payload={
                        "text": chunks[i]["text"],
                        "source": chunks[i]["source"],
                        "page": chunks[i]["page"],
                        "chunk": chunks[i]["chunk"]
                    }
                )
                for i in range(len(chunks))
            ]
            
            # Upload to Qdrant
            logger.info(f"Uploading {len(points)} points to Qdrant collection {collection_name}")
            client.upsert(collection_name=collection_name, points=points)
            
            logger.info(f"Successfully processed and uploaded {file.filename}")
            return {
                "status": "success", 
                "message": f"Uploaded {len(points)} chunks from {file.filename}.",
                "chunks_count": len(points)
            }
        
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                logger.info(f"Removing temporary file: {temp_file_path}")
                os.unlink(temp_file_path)
    
    except Exception as e:
        logger.error(f"Error in process_pdf: {str(e)}")
        raise 