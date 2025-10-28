import logging
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Union, Optional
import uvicorn

from app.models.schemas import (
	UpdateRequest, SaveResponse, UpdateResponse, ErrorResponse,
	GeoSearchRequest, Business, Service, Product
)
from app.api.handlers import DataAPIHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
	title="Sangmaner AI Data Storage API",
	description="Data storage and management system for Sangmaner AI search engine",
	version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # Configure appropriately for production
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Initialize API handler
api_handler = DataAPIHandler()


@app.get("/")
async def root():
	"""Health check endpoint"""
	return {"message": "Sangmaner AI Data Storage API is running"}


@app.get("/health")
async def health_check():
	"""Detailed health check"""
	try:
		# Test Qdrant connection using safe calls (avoid strict model parsing)
		client = api_handler.qdrant_manager.client
		collections = client.get_collections()
		collection_name = api_handler.qdrant_manager.collection_name
		points_count = None
		try:
			count_resp = client.count(collection_name=collection_name, exact=True)
			# Some client versions return object with "count" attribute, fallback to dict
			points_count = getattr(count_resp, "count", None) or (count_resp.get("count") if isinstance(count_resp, dict) else None)
		except Exception:
			points_count = None

		return {
			"status": "healthy",
			"qdrant_connection": "ok",
			"collection_name": collection_name,
			"points_count": points_count,
			"message": "All systems operational"
		}
	except Exception as e:
		logger.error(f"Health check failed: {e}")
		raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/data/save", response_model=SaveResponse)
async def save_data(request: Union[Business, Service, Product]):
	"""
	Save new business, service, or product data to Qdrant.
	
	This endpoint supports upsert operations - if an entity with the same ID exists,
	it will be updated with the new data.
	
	**Business Example:**
	```json
	{
		"business_id": 0,
		"name": "string",
		"description": "string",
		"business_type": "string",
		"user_id": "string",
		"client_id": 0,
		"status": "active",
		"pincode": "string",
		"city": "string",
		"state": "string",
		"country": "string",
		"phone": "string",
		"email": "string",
		"website": "string",
		"lat": 0,
		"long": 0,
		"tags": []
	}
	```
	
	**Service Example:**
	```json
	{
		"service_id": "string",
		"service_name": "string",
		"description": "string",
		"business_id": "string",
		"client_id": "string",
		"user_id": "string",
		"lat": 0,
		"long": 0,
		"tags": []
	}
	```
	
	**Product Example:**
	```json
	{
		"product_id": "string",
		"product_name": "string",
		"description": "string",
		"business_id": "string",
		"client_id": "string",
		"lat": 0,
		"long": 0,
		"tags": []
	}
	```
	"""
	return await api_handler.save_data(request)


@app.put("/data/update/{entity_id}", response_model=UpdateResponse)
async def update_data(entity_id: str, update_request: UpdateRequest):
	"""
	Update existing entity data in Qdrant.
	
	Only provided fields will be updated. If textual fields (name, description, tags)
	are changed, the embedding will be automatically regenerated.
	
	**Example:**
	```json
	{
		"business_name": "Sharma Tea Stall & Snacks",
		"address": "New MG Road, Sangmaner",
		"phone": "9876500000"
	}
	```
	"""
	return await api_handler.update_data(entity_id, update_request)


@app.get("/data/{entity_id}")
async def get_entity(entity_id: str):
	"""
	Get entity by ID.
	
	Returns the complete entity data including vector embedding and payload.
	"""
	return await api_handler.get_entity(entity_id)


@app.delete("/data/{entity_id}")
async def delete_entity(entity_id: str):
	"""
	Delete entity by ID.
	
	**Warning:** This operation is irreversible. Consider the impact on related
	services and products before deleting a business.
	"""
	return await api_handler.delete_entity(entity_id)


@app.post("/data/search/nearby")
async def search_nearby_services(geo_request: GeoSearchRequest):
	"""
	Search for nearby entities (businesses, services, products) for a client using semantic + distance sort.
	
	The server applies an increasing geo radius internally (2km → 5km → 10km → 20km → ...)
	until matches are found, then returns results with the full stored payload for each entity.
	
	**Request Body:**
	```json
	{
		"latitude": 19.123,
		"longitude": 73.456,
		"client_id": "client-1",
		"query": "tea breakfast"
	}
	```
	
	**Response:**
	Returns mixed entities with fields: entity_type, entity_id, score, payload (entire stored object).
	"""
	return await api_handler.geo_search_services(geo_request)


@app.post("/data/ingest/pdf")
async def ingest_pdf_endpoint(
	client_id: str = Form(...),
	file: UploadFile = File(...)
):
	"""Upload a PDF and ingest it into the taluka collection for the given client_id.

	Form-data fields:
	- client_id: string
	- file: PDF file
	"""
	if not file.filename.lower().endswith(".pdf"):
		raise HTTPException(status_code=400, detail="Only PDF files are supported")
	content = await file.read()
	return await api_handler.ingest_pdf(client_id=client_id, file_bytes=content, filename=file.filename)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
	"""Custom HTTP exception handler"""
	from fastapi.responses import JSONResponse
	return JSONResponse(
		status_code=exc.status_code,
		content={
			"status": "error",
			"message": exc.detail,
			"details": {"status_code": exc.status_code}
		}
	)


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
	"""General exception handler"""
	from fastapi.responses import JSONResponse
	logger.error(f"Unhandled exception: {exc}")
	return JSONResponse(
		status_code=500,
		content={
			"status": "error",
			"message": "Internal server error",
			"details": {"error": str(exc)}
		}
	)


if __name__ == "__main__":
	uvicorn.run(
		"main:app",
		host="0.0.0.0",
		port=8001,
		reload=True,
		log_level="info",
		log_config={
			"version": 1,
			"disable_existing_loggers": False,
			"formatters": {
				"default": {
					"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
				}
			},
			"handlers": {
				"default": {
					"class": "logging.StreamHandler",
					"stream": "ext://sys.stdout",
					"formatter": "default",
				}
			},
			"root": {
				"handlers": ["default"],
				"level": "INFO"
			}
		}
	)
