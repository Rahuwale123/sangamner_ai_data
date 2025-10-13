import logging
import uuid
from typing import List, Dict, Any, Optional, Union
from qdrant_client import QdrantClient as QdrantDBClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import json

from app.core.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME, EMBEDDING_MODEL, VECTOR_SIZE
from app.models.schemas import BusinessPayload, ServicePayload, ProductPayload, Location

logger = logging.getLogger(__name__)


class QdrantManager:
	def __init__(self):
		self.client = QdrantDBClient(host=QDRANT_HOST, port=QDRANT_PORT)
		self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
		self.collection_name = QDRANT_COLLECTION_NAME
		self._ensure_collection_exists()

	def _ensure_collection_exists(self):
		"""Create collection if it doesn't exist"""
		try:
			collections = self.client.get_collections()
			collection_names = [col.name for col in collections.collections]
			
			if self.collection_name not in collection_names:
				self.client.create_collection(
					collection_name=self.collection_name,
					vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
				)
				logger.info(f"Created collection: {self.collection_name}")
			else:
				logger.info(f"Collection {self.collection_name} already exists")
		except Exception as e:
			logger.error(f"Error ensuring collection exists: {e}")
			raise

	def _generate_embedding(self, text: str) -> List[float]:
		"""Generate embedding for given text"""
		try:
			embedding = self.embedding_model.encode(text)
			return embedding.tolist()
		except Exception as e:
			logger.error(f"Error generating embedding: {e}")
			raise

	def _string_to_uuid(self, string_id: str) -> str:
		"""Convert string ID to UUID format for Qdrant"""
		return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))

	def _extract_text_for_embedding(self, payload: Union[BusinessPayload, ServicePayload, ProductPayload]) -> str:
		"""Extract text fields for embedding generation"""
		text_parts = []
		
		if hasattr(payload, 'business_name'):
			text_parts.append(payload.business_name)
		if hasattr(payload, 'service_name'):
			text_parts.append(payload.service_name)
		if hasattr(payload, 'product_name'):
			text_parts.append(payload.product_name)
		
		if payload.description:
			text_parts.append(payload.description)
		
		if payload.tags:
			text_parts.extend(payload.tags)
		
		return " ".join(text_parts)

	def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
		"""Calculate distance between two points using Haversine formula"""
		import math
		
		lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
		
		dlat = lat2 - lat1
		dlon = lon2 - lon1
		a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
		c = 2 * math.asin(math.sqrt(a))
		
		r = 6371
		return c * r

	def save_business(self, payload: BusinessPayload) -> str:
		"""Save business to Qdrant"""
		try:
			text = self._extract_text_for_embedding(payload)
			vector = self._generate_embedding(text)
			
			qdrant_payload = payload.model_dump()
			qdrant_payload['location'] = payload.location.model_dump()
			
			point = PointStruct(
				id=self._string_to_uuid(payload.business_id),
				vector=vector,
				payload=qdrant_payload
			)
			
			self.client.upsert(
				collection_name=self.collection_name,
				points=[point]
			)
			
			logger.info(f"Saved business: {payload.business_id}")
			return payload.business_id
			
		except Exception as e:
			logger.error(f"Error saving business: {e}")
			raise

	def save_service(self, payload: ServicePayload) -> str:
		"""Save service to Qdrant"""
		try:
			text = self._extract_text_for_embedding(payload)
			vector = self._generate_embedding(text)
			
			qdrant_payload = payload.model_dump()
			qdrant_payload['location'] = payload.location.model_dump()
			
			point = PointStruct(
				id=self._string_to_uuid(payload.service_id),
				vector=vector,
				payload=qdrant_payload
			)
			
			self.client.upsert(
				collection_name=self.collection_name,
				points=[point]
			)
			
			logger.info(f"Saved service: {payload.service_id}")
			return payload.service_id
			
		except Exception as e:
			logger.error(f"Error saving service: {e}")
			raise

	def save_product(self, payload: ProductPayload) -> str:
		"""Save product to Qdrant"""
		try:
			text = self._extract_text_for_embedding(payload)
			vector = self._generate_embedding(text)
			
			qdrant_payload = payload.model_dump()
			qdrant_payload['location'] = payload.location.model_dump()
			
			point = PointStruct(
				id=self._string_to_uuid(payload.product_id),
				vector=vector,
				payload=qdrant_payload
			)
			
			self.client.upsert(
				collection_name=self.collection_name,
				points=[point]
			)
			
			logger.info(f"Saved product: {payload.product_id}")
			return payload.product_id
			
		except Exception as e:
			logger.error(f"Error saving product: {e}")
			raise

	def get_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
		"""Get entity by ID"""
		try:
			points = self.client.retrieve(
				collection_name=self.collection_name,
				ids=[self._string_to_uuid(entity_id)]
			)
			
			if points:
				point = points[0]
				return {
					'id': point.id,
					'vector': point.vector,
					'payload': point.payload
				}
			
			return None
			
		except Exception as e:
			logger.error(f"Error getting entity by ID: {e}")
			return None

	def update_entity(self, entity_id: str, update_data: Dict[str, Any]) -> str:
		"""Update existing entity"""
		try:
			existing = self.get_by_id(entity_id)
			if not existing:
				raise ValueError(f"Entity with ID {entity_id} not found")
			
			updated_payload = existing['payload'].copy()
			updated_payload.update(update_data)
			
			text_fields = ['business_name', 'service_name', 'product_name', 'description', 'tags']
			text_changed = any(field in update_data for field in text_fields)
			
			if text_changed:
				text = self._extract_text_for_embedding_from_payload(updated_payload)
				vector = self._generate_embedding(text)
			else:
				vector = existing['vector']
			
			point = PointStruct(
				id=self._string_to_uuid(entity_id),
				vector=vector,
				payload=updated_payload
			)
			
			self.client.upsert(
				collection_name=self.collection_name,
				points=[point]
			)
			
			logger.info(f"Updated entity: {entity_id}")
			return entity_id
			
		except Exception as e:
			logger.error(f"Error updating entity: {e}")
			raise

	def _extract_text_for_embedding_from_payload(self, payload: Dict[str, Any]) -> str:
		"""Extract text fields for embedding generation from payload dict"""
		text_parts = []
		
		for name_field in ['business_name', 'service_name', 'product_name']:
			if name_field in payload and payload[name_field]:
				text_parts.append(payload[name_field])
		
		if 'description' in payload and payload['description']:
			text_parts.append(payload['description'])
		
		if 'tags' in payload and payload['tags']:
			text_parts.extend(payload['tags'])
		
		return " ".join(text_parts)

	def delete_entity(self, entity_id: str) -> bool:
		"""Delete entity by ID"""
		try:
			self.client.delete(
				collection_name=self.collection_name,
				points_selector=[self._string_to_uuid(entity_id)]
			)
			
			logger.info(f"Deleted entity: {entity_id}")
			return True
			
		except Exception as e:
			logger.error(f"Error deleting entity: {e}")
			return False

	def geo_search_entities(self, latitude: float, longitude: float, client_id: Union[int, str], query: str) -> List[Dict[str, Any]]:
		"""Search for entities (business/service/product) for a client_id using semantic search, sorted by combined semantic+geo score.
		Applies an increasing geo radius until results are found, then ranks within that radius.
		"""
		try:
			from qdrant_client.models import GeoPoint, GeoRadius

			query_vector = self._generate_embedding(query)
			query_lower_tokens = set(str(query).lower().split())

			search_radii_meters = [2000, 5000, 10000, 20000, 50000, 100000, 200000]

			best_results: List[Dict[str, Any]] = []
			used_radius_m: Optional[int] = None

			ABS_MIN_RAW_SIM = 0.35

			for radius_m in search_radii_meters:
				client_values: List[Union[str, int]] = []
				try:
					client_values.append(str(client_id))
					client_values.append(int(str(client_id)))
				except Exception:
					client_values.append(str(client_id))

				client_should = [
					FieldCondition(key='client_id', match=MatchValue(value=v))
					for v in client_values
				]

				must_conditions = [
					FieldCondition(
						key='location',
						geo_radius=GeoRadius(
							center=GeoPoint(lat=latitude, lon=longitude),
							radius=radius_m
						)
					),
				]

				search_filter = Filter(must=must_conditions, should=client_should)

				search_results = self.client.search(
					collection_name=self.collection_name,
					query_vector=query_vector,
					limit=1000,
					query_filter=search_filter
				)

				if not search_results:
					continue

				interim: List[Dict[str, Any]] = []

				for res in search_results:
					payload = res.payload or {}
					loc = payload.get('location') or {}
					if not isinstance(loc, dict) or 'lat' not in loc or 'lon' not in loc:
						continue

					entity_lat = float(loc['lat'])
					entity_lon = float(loc['lon'])
					distance_km = self._calculate_distance(latitude, longitude, entity_lat, entity_lon)

					cleaned_payload = {k: v for k, v in dict(payload).items() if v is not None}
					loc = cleaned_payload.get('location')
					if isinstance(loc, dict):
						if 'lat' in loc:
							cleaned_payload['lat'] = loc.get('lat')
						if 'lon' in loc:
							cleaned_payload['long'] = loc.get('lon')
						cleaned_payload.pop('location', None)
					if cleaned_payload.get('description') is None:
						cleaned_payload['description'] = ""
					if cleaned_payload.get('tags') is None:
						cleaned_payload['tags'] = []

					raw_score = float(res.score or 0.0)

					tags = [str(t).lower() for t in (cleaned_payload.get('tags') or [])]
					names = [
						str(cleaned_payload.get('product_name', '')).lower(),
						str(cleaned_payload.get('service_name', '')).lower(),
						str(cleaned_payload.get('business_name', '')).lower(),
						str(cleaned_payload.get('description', '')).lower(),
					]
					text_tokens = set((" ".join(tags + names)).split())
					has_keyword_overlap = len(query_lower_tokens.intersection(text_tokens)) > 0

					if not has_keyword_overlap and raw_score < ABS_MIN_RAW_SIM:
						continue

					interim.append({
						'qdrant_id': res.id,
						'raw_score': raw_score,
						'distance_km': round(distance_km, 6),
						'payload': cleaned_payload,
						'has_keyword_overlap': has_keyword_overlap
					})

				if not interim:
					continue

				results_scored: List[Dict[str, Any]] = []

				for item in interim:
					semantic_norm = max(0.0, min(1.0, item['raw_score']))

					if semantic_norm < 0.4 and not item['has_keyword_overlap']:
						continue

					distance_score = 1.0 - min((item['distance_km'] * 1000.0) / float(radius_m), 1.0)

					payload = item['payload']
					entity_type = payload.get('type')

					keyword_boost = 0.1 if item['has_keyword_overlap'] else 0.0

					if entity_type == 'product':
						type_boost = 0.15
					elif entity_type == 'service':
						type_boost = 0.08
					else:
						type_boost = 0.0

					combined_score = 0.7 * semantic_norm + 0.3 * distance_score + type_boost + keyword_boost

					if entity_type == 'business':
						domain_id = payload.get('business_id')
					elif entity_type == 'service':
						domain_id = payload.get('service_id')
					elif entity_type == 'product':
						domain_id = payload.get('product_id')
					else:
						domain_id = None

					if not domain_id:
						continue

					results_scored.append({
						'entity_id': domain_id,
						'entity_type': entity_type,
						'score': round(semantic_norm, 4),
						'distance_km': round(item['distance_km'], 2),
						'payload': payload
					})

				if results_scored:
					results_scored.sort(key=lambda x: (-x['score'], x.get('distance_km', 0)))
					best_results = results_scored
					used_radius_m = radius_m
					break

			return best_results

		except Exception as e:
			logger.error(f"Error in geo search: {e}")
			return []

	def search_similar(self, query: str, limit: int = 10, filter_conditions: Optional[Dict] = None) -> List[Dict[str, Any]]:
		"""Search for similar entities"""
		try:
			query_vector = self._generate_embedding(query)
			
			search_filter = None
			if filter_conditions:
				conditions = []
				for key, value in filter_conditions.items():
					conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
				search_filter = Filter(must=conditions)
			
			results = self.client.search(
				collection_name=self.collection_name,
				query_vector=query_vector,
				limit=limit,
				query_filter=search_filter
			)
			
			return [
				{
					'id': result.id,
					'score': result.score,
					'payload': result.payload
				}
				for result in results
			]
			
		except Exception as e:
			logger.error(f"Error searching similar entities: {e}")
			return []
