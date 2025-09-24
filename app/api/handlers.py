import logging
from typing import Union, Dict, Any
from fastapi import HTTPException

from app.models.schemas import (
	UpdateRequest, SaveResponse, UpdateResponse, ErrorResponse,
	BusinessPayload, ServicePayload, ProductPayload, EntityType,
	GeoSearchRequest, Business, Service, Product, Location
)
from app.services.qdrant_manager import QdrantManager
from app.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)


class DataAPIHandler:
	def __init__(self):
		self.qdrant_manager = QdrantManager()
		self.gemini_service = GeminiService()

	async def save_data(self, request: Union[Business, Service, Product]) -> SaveResponse:
		"""Handle save data API request"""
		try:
			# Determine entity by presence of id/name fields
			if isinstance(request, Business):
				return await self._save_business(request)
			elif isinstance(request, Service):
				return await self._save_service(request)
			elif isinstance(request, Product):
				return await self._save_product(request)
			else:
				raise HTTPException(status_code=400, detail="Invalid entity payload")
				
		except HTTPException:
			raise
		except Exception as e:
			logger.error(f"Error in save_data: {e}")
			raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

	async def _save_business(self, request: Business) -> SaveResponse:
		"""Save business entity"""
		try:
			# Map new schema to payload
			location = Location(lat=request.lat, lon=request.long)
			payload = BusinessPayload(
				business_id=str(request.business_id),
				business_name=request.name,
				group_id=None,
				location=location,
				address=None,
				phone=request.phone,
				description=request.description,
				tags=request.tags or [],
				client_id=str(request.client_id),
				user_id=request.user_id,
				business_type=request.business_type,
				status=request.status,
				pincode=request.pincode,
				city=request.city,
				state=request.state,
				country=request.country,
				email=request.email,
				website=request.website,
			)
			
			# Save to Qdrant
			entity_id = self.qdrant_manager.save_business(payload)
			
			return SaveResponse(status="success", id=entity_id)
			
		except Exception as e:
			logger.error(f"Error saving business: {e}")
			raise HTTPException(status_code=400, detail=f"Error saving business: {str(e)}")

	async def _save_service(self, request: Service) -> SaveResponse:
		"""Save service entity"""
		try:
			# Validate business exists only if provided
			if request.business_id:
				business = self.qdrant_manager.get_by_id(request.business_id)
				if not business or business['payload'].get('type') != 'business':
					raise HTTPException(status_code=400, detail=f"Business with ID {request.business_id} not found")
			
			# Map new schema to payload
			location = Location(lat=request.lat, lon=request.long)
			payload = ServicePayload(
				service_id=request.service_id,
				business_id=request.business_id,
				service_name=request.service_name,
				group_id=None,
				location=location,
				price=None,
				description=request.description,
				tags=request.tags or [],
				client_id=request.client_id,
				user_id=request.user_id,
			)
			
			# Save to Qdrant
			entity_id = self.qdrant_manager.save_service(payload)
			
			return SaveResponse(status="success", id=entity_id)
			
		except HTTPException:
			raise
		except Exception as e:
			logger.error(f"Error saving service: {e}")
			raise HTTPException(status_code=400, detail=f"Error saving service: {str(e)}")

	async def _save_product(self, request: Product) -> SaveResponse:
		"""Save product entity"""
		try:
			# Validate business exists if provided
			if request.business_id:
				business = self.qdrant_manager.get_by_id(request.business_id)
				if not business or business['payload'].get('type') != 'business':
					raise HTTPException(status_code=400, detail=f"Business with ID {request.business_id} not found")
			
			# Validate service exists if provided (product schema doesn't carry it, so skip)
			
			# Map new schema to payload
			location = Location(lat=request.lat, lon=request.long)
			payload = ProductPayload(
				product_id=request.product_id,
				service_id=None,
				business_id=request.business_id if request.business_id else "",
				product_name=request.product_name,
				group_id=None,
				location=location,
				price=None,
				description=request.description,
				tags=request.tags or [],
				client_id=request.client_id,
			)
			
			# Save to Qdrant
			entity_id = self.qdrant_manager.save_product(payload)
			
			return SaveResponse(status="success", id=entity_id)
			
		except HTTPException:
			raise
		except Exception as e:
			logger.error(f"Error saving product: {e}")
			raise HTTPException(status_code=400, detail=f"Error saving product: {str(e)}")

	async def update_data(self, entity_id: str, update_request: UpdateRequest) -> UpdateResponse:
		"""Handle update data API request"""
		try:
			# Get existing entity
			existing = self.qdrant_manager.get_by_id(entity_id)
			if not existing:
				raise HTTPException(status_code=404, detail=f"Entity with ID {entity_id} not found")
			
			# Prepare update data (only non-None values)
			update_data = {}
			for field, value in update_request.model_dump().items():
				if value is not None:
					if field == 'location' and isinstance(value, dict):
						update_data[field] = value
					else:
						update_data[field] = value
			
			if not update_data:
				raise HTTPException(status_code=400, detail="No fields to update")
			
			# Update entity
			updated_id = self.qdrant_manager.update_entity(entity_id, update_data)
			
			return UpdateResponse(status="updated", id=updated_id)
			
		except HTTPException:
			raise
		except Exception as e:
			logger.error(f"Error updating entity: {e}")
			raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

	async def get_entity(self, entity_id: str) -> Dict[str, Any]:
		"""Get entity by ID"""
		try:
			entity = self.qdrant_manager.get_by_id(entity_id)
			if not entity:
				raise HTTPException(status_code=404, detail=f"Entity with ID {entity_id} not found")
			
			return entity
			
		except HTTPException:
			raise
		except Exception as e:
			logger.error(f"Error getting entity: {e}")
			raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

	async def delete_entity(self, entity_id: str) -> Dict[str, str]:
		"""Delete entity by ID"""
		try:
			# Check if entity exists
			existing = self.qdrant_manager.get_by_id(entity_id)
			if not existing:
				raise HTTPException(status_code=404, detail=f"Entity with ID {entity_id} not found")
			
			# Delete entity
			success = self.qdrant_manager.delete_entity(entity_id)
			if not success:
				raise HTTPException(status_code=500, detail="Failed to delete entity")
			
			return {"status": "deleted", "id": entity_id}
			
		except HTTPException:
			raise
		except Exception as e:
			logger.error(f"Error deleting entity: {e}")
			raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

	async def geo_search_services(self, geo_request: GeoSearchRequest) -> Dict[str, Any]:
		"""Search across client using semantic + distance sort (iterative geo radius, hidden from client)."""
		try:
			results = self.qdrant_manager.geo_search_entities(
				latitude=geo_request.latitude,
				longitude=geo_request.longitude,
				client_id=geo_request.client_id,
				query=geo_request.query
			)
			
			response_data = {
				"status": "success",
				"search_params": {
					"latitude": geo_request.latitude,
					"longitude": geo_request.longitude,
					"client_id": geo_request.client_id,
					"query": geo_request.query,
					"ai_mode": geo_request.ai_mode
				},
				"results": results,
				"total": len(results)
			}
			
			# Add AI response if AI mode is enabled
			if geo_request.ai_mode and results:
				try:
					# Prepare location context for AI
					location_context = f"Sangamner area (coordinates: {geo_request.latitude}, {geo_request.longitude})"
					
					# Generate AI summary
					ai_response = await self.gemini_service.generate_search_summary(
						search_results=results,
						query=geo_request.query,
						location=location_context
					)
					
					response_data["ai_response"] = ai_response
					
				except Exception as e:
					logger.error(f"Error generating AI response: {e}")
					response_data["ai_response"] = "AI summary is currently unavailable. Please try again later."
			elif geo_request.ai_mode and not results:
				response_data["ai_response"] = "No results found for your search query. Try adjusting your search terms or location."
			
			return response_data
			
		except Exception as e:
			logger.error(f"Error in geo search: {e}")
			raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

	async def search_entities(self, query: str, limit: int = 10, entity_type: str = None, group_id: int = None) -> Dict[str, Any]:
		"""Search for entities"""
		try:
			# Build filter conditions
			filter_conditions = {}
			if entity_type:
				filter_conditions['type'] = entity_type
			if group_id is not None:
				filter_conditions['group_id'] = group_id
			
			# Search
			results = self.qdrant_manager.search_similar(
				query=query,
				limit=limit,
				filter_conditions=filter_conditions if filter_conditions else None
			)
			
			return {
				"status": "success",
				"query": query,
				"results": results,
				"total": len(results)
			}
			
		except Exception as e:
			logger.error(f"Error searching entities: {e}")
			raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
