import os
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from api.pdf_upload import process_pdf
from api.search import semantic_search
from api.nutrition import get_nutrition_recommendation
from api.logging_utils import setup_logger
from pydantic import BaseModel

# Set up logger
logger = setup_logger(__name__)

# Load environment variables
logger.info("Loading environment variables")
load_dotenv()

app = FastAPI(
    title="PDF Semantic Search API",
    description="API for uploading PDFs and performing semantic search on their content",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please check the logs for more details."}
    )

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to PDF Semantic Search API"}

@app.post("/upload-pdf/")
async def upload_pdf_endpoint(
    file: UploadFile = File(...),
    collection_name: str = Form("health_schemes")
):
    """
    Upload a PDF file, process it, and store embeddings in Qdrant
    """
    logger.info(f"Upload PDF endpoint accessed: {file.filename}, collection: {collection_name}")
    
    if not file.filename.endswith('.pdf'):
        logger.warning(f"Invalid file format: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        result = await process_pdf(file, collection_name)
        logger.info(f"PDF upload successful: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in upload_pdf_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class SearchRequest(BaseModel):
    query: str
    collection_name: str = "health_schemes"
    limit: int = 3

class NutritionRequest(BaseModel):
    query: str
    limit: int = 3

@app.post("/search/")
async def search_endpoint(
    request: SearchRequest
):
    """
    Perform semantic search on stored PDF content
    """
    logger.info(f"Search endpoint accessed: query='{request.query}', collection={request.collection_name}, limit={request.limit}")
    
    try:
        results = await semantic_search(request.query, request.collection_name, request.limit)
        logger.info(f"Search successful: {len(results['results'])} results found")
        return results
    except Exception as e:
        logger.error(f"Error in search_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/nutrition-recommendation/")
async def nutrition_recommendation_endpoint(
    request: NutritionRequest
):
    """
    Get nutrition recommendations for pregnant women based on their details
    and generate a weekly diet plan tailored to their geographical region
    """
    logger.info(f"Nutrition recommendation endpoint accessed: query='{request.query}', limit={request.limit}")
    
    try:
        recommendation = await get_nutrition_recommendation(request.query, request.limit)
        logger.info("Nutrition recommendation generated successfully")
        return recommendation
    except Exception as e:
        logger.error(f"Error in nutrition_recommendation_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting application")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 