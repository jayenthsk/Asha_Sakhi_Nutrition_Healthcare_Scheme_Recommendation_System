# ASHA SAKHI Backend Health Scheme Recommendation, and Nutrition Recommendation System

A FastAPI application for uploading PDFs, processing them, and performing semantic search on their content.

## Setup

1. Clone this repository
2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file based on `.env.example` and fill in your Qdrant credentials and Llama API key
5. Download the embedding model:
   ```
   python download_model.py
   ```
6. Run the application:
   ```
   uvicorn app:app --reload
   ```

## API Endpoints

### Upload PDF
- **URL**: `/upload-pdf/`
- **Method**: `POST`
- **Form Data**:
  - `file`: PDF file
  - `collection_name`: (Optional) Name of the Qdrant collection (default: "health_schemes")

### Search
- **URL**: `/search/`
- **Method**: `GET`
- **Query Parameters**:
  - `query`: Search query
  - `collection_name`: (Optional) Name of the Qdrant collection (default: "health_schemes")
  - `limit`: (Optional) Maximum number of results to return (default: 3)

### Nutrition Recommendation
- **URL**: `/nutrition-recommendation/`
- **Method**: `POST`
- **Query Parameters**:
  - `query`: Details of the pregnant woman including geographical region
  - `limit`: (Optional) Maximum number of nutrition information results to retrieve (default: 3)

## Directory Structure

.
├── app.py                # Main FastAPI application
├── api/
│   ├── pdf_upload.py     # PDF processing and upload logic
│   ├── search.py         # Semantic search logic
│   └── nutrition.py      # Nutrition recommendation logic
├── models/               # Directory for storing the embedding model
├── requirements.txt      # Python dependencies
└── download_model.py     # Script to download the embedding model 
