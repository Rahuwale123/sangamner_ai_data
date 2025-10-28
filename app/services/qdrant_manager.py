import logging
import uuid
from typing import List, Dict, Any, Optional, Union
from qdrant_client import QdrantClient as QdrantDBClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import json

from app.core.config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME, EMBEDDING_MODEL, VECTOR_SIZE,
    SEMANTIC_SCORE_THRESHOLD, SEMANTIC_WEIGHT, GEO_WEIGHT, TALUKA_COLLECTION_NAME,
)
from app.models.schemas import BusinessPayload, ServicePayload, ProductPayload, Location

logger = logging.getLogger(__name__)


class QdrantManager:
	def __init__(self):
		# Try server mode first, fall back to local path if fails
		try:
			self.client = QdrantDBClient(host=QDRANT_HOST, port=QDRANT_PORT)
			logger.info(f"Connected to Qdrant server at {QDRANT_HOST}:{QDRANT_PORT}")
		except Exception as e:
			logger.warning(f"Failed to connect to Qdrant server: {e}. Trying local path...")
			try:
				self.client = QdrantDBClient(path="./qdrant_data")
				logger.info("Connected to Qdrant using local path: ./qdrant_data")
			except Exception as e2:
				logger.error(f"Failed to connect to Qdrant using local path: {e2}")
				raise
		
		self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
		self.collection_name = QDRANT_COLLECTION_NAME
		self._ensure_collection_exists()

	def _ensure_collection_exists_by_name(self, collection_name: str):
		try:
			collections = self.client.get_collections()
			collection_names = [col.name for col in collections.collections]
			if collection_name not in collection_names:
				self.client.create_collection(
					collection_name=collection_name,
					vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
				)
				logger.info(f"Created collection: {collection_name}")
		except Exception as e:
			logger.error(f"Error ensuring collection {collection_name}: {e}")
			raise

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
		
		# Names get moderate weight
		if hasattr(payload, 'business_name') and payload.business_name:
			text_parts.extend([payload.business_name] * 2)
		if hasattr(payload, 'service_name') and payload.service_name:
			text_parts.extend([payload.service_name] * 2)
		if hasattr(payload, 'product_name') and payload.product_name:
			text_parts.extend([payload.product_name] * 2)
		
		# Description as-is
		if getattr(payload, 'description', None):
			text_parts.append(payload.description)
		
		# Location semantics (city/state/country/pincode) to help geographic queries
		for loc_field in ['city', 'state', 'country', 'pincode']:
			val = getattr(payload, loc_field, None)
			if val:
				text_parts.extend([str(val)] * 2)
		
		# Tags are highly important for retrieval; upweight by repeating
		tags = getattr(payload, 'tags', []) or []
		if tags:
			lower_tags = [str(t).strip().lower() for t in tags if t]
			# Repeat tags to upweight them without hardcoded synonyms
			text_parts.extend(lower_tags * 3)
		
		return " ".join([str(p) for p in text_parts if p])

	def _chunk_text_by_words(self, text: str, chunk_size_words: int = 200, overlap_words: int = 50) -> List[str]:
		words = [w for w in (text or "").split() if w]
		if not words:
			return []
		chunks: List[str] = []
		start = 0
		while start < len(words):
			end = min(start + chunk_size_words, len(words))
			chunk = " ".join(words[start:end])
			if chunk.strip():
				chunks.append(chunk)
			if end == len(words):
				break
			start = end - overlap_words
			if start < 0:
				start = 0
		return chunks

	def ingest_document(self, client_id: Union[str, int], file_bytes: bytes, filename: str, chunk_size_words: int = 200) -> int:
		"""Ingest a document (PDF/TXT/DOC): extract text, chunk, embed, and upsert into TALUKA collection.
		Returns the number of chunks indexed.
		"""
		try:
			# Ensure collection exists
			self._ensure_collection_exists_by_name(TALUKA_COLLECTION_NAME)

			# Extract text based on file type
			full_text = ""
			file_ext = filename.lower().split('.')[-1]
			
			if file_ext == 'pdf':
				full_text = self._extract_text_from_pdf(file_bytes)
			elif file_ext == 'txt':
				full_text = self._extract_text_from_txt(file_bytes)
			elif file_ext in ['doc', 'docx']:
				full_text = self._extract_text_from_doc(file_bytes, file_ext)
			elif file_ext == 'csv':
				full_text = self._extract_text_from_csv(file_bytes)
			else:
				raise ValueError(f"Unsupported file type: {file_ext}. Supported types: PDF, TXT, DOC, DOCX, CSV")

			if not full_text or not full_text.strip():
				logger.warning(f"No text extracted from {filename}")
				return 0

			# Chunk
			chunks = self._chunk_text_by_words(full_text, chunk_size_words=chunk_size_words)
			if not chunks:
				return 0

			# Prepare and upsert
			points: List[PointStruct] = []
			client_id_str = str(client_id)
			for idx, chunk in enumerate(chunks):
				try:
					vector = self._generate_embedding(chunk)
				except Exception as e:
					logger.warning(f"Embedding failed for chunk {idx}: {e}")
					continue
				payload: Dict[str, Any] = {
					"type": "document_chunk",
					"client_id": client_id_str,
					"filename": filename,
					"page_chunks": len(chunks),
					"chunk_index": idx,
					"text": chunk,
				}
				point = PointStruct(
					id=self._string_to_uuid(f"{client_id_str}:{filename}:{idx}"),
					vector=vector,
					payload=payload
				)
				points.append(point)

			if not points:
				return 0

			self.client.upsert(
				collection_name=TALUKA_COLLECTION_NAME,
				points=points
			)
			logger.info(f"Ingested {len(points)} chunks from {filename} into {TALUKA_COLLECTION_NAME} for client_id={client_id_str}")
			return len(points)
			
		except Exception as e:
			logger.error(f"Error ingesting document {filename}: {e}")
			raise

	def _extract_text_from_pdf(self, file_bytes: bytes) -> str:
		"""Extract text from PDF file."""
		try:
			from PyPDF2 import PdfReader
			import io
			
			reader = PdfReader(io.BytesIO(file_bytes))
			pages_text: List[str] = []
			for page in reader.pages:
				try:
					pages_text.append(page.extract_text() or "")
				except Exception:
					pages_text.append("")
			return "\n".join(pages_text)
		except Exception as e:
			logger.error(f"Failed to extract text from PDF: {e}")
			raise

	def _extract_text_from_txt(self, file_bytes: bytes) -> str:
		"""Extract text from TXT file."""
		try:
			# Try UTF-8 first, fall back to other encodings
			try:
				return file_bytes.decode('utf-8')
			except UnicodeDecodeError:
				try:
					return file_bytes.decode('latin-1')
				except UnicodeDecodeError:
					return file_bytes.decode('cp1252')
		except Exception as e:
			logger.error(f"Failed to extract text from TXT: {e}")
			raise

	def _extract_text_from_doc(self, file_bytes: bytes, file_ext: str) -> str:
		"""Extract text from DOC/DOCX file."""
		try:
			import io
			
			if file_ext == 'docx':
				try:
					from docx import Document
					doc = Document(io.BytesIO(file_bytes))
					text_parts = []
					for paragraph in doc.paragraphs:
						if paragraph.text.strip():
							text_parts.append(paragraph.text)
					return "\n".join(text_parts)
				except ImportError:
					logger.error("python-docx not installed. Install with: pip install python-docx")
					raise
			else:  # .doc file
				try:
					import textract
					return textract.process(io.BytesIO(file_bytes)).decode('utf-8')
				except ImportError:
					logger.error("textract not installed. Install with: pip install textract")
					raise
		except Exception as e:
			logger.error(f"Failed to extract text from {file_ext.upper()}: {e}")
			raise

	def _extract_text_from_csv(self, file_bytes: bytes) -> str:
		"""Extract text from CSV file."""
		try:
			import csv
			import io
			
			# Try to decode with different encodings
			try:
				text_data = file_bytes.decode('utf-8')
			except UnicodeDecodeError:
				try:
					text_data = file_bytes.decode('latin-1')
				except UnicodeDecodeError:
					text_data = file_bytes.decode('cp1252')
			
			# Parse CSV
			csv_reader = csv.reader(io.StringIO(text_data))
			rows = list(csv_reader)
			
			if not rows:
				return ""
			
			# Convert CSV to readable text format
			text_parts = []
			headers = rows[0] if rows else []
			
			# Add headers
			if headers:
				text_parts.append("Headers: " + ", ".join(headers))
				text_parts.append("")
			
			# Add rows in readable format
			for idx, row in enumerate(rows[1:], 1):
				if not any(cell.strip() for cell in row):  # Skip empty rows
					continue
				
				row_text = f"Row {idx}: "
				for header, value in zip(headers, row):
					if value.strip():
						row_text += f"{header}={value}, "
				text_parts.append(row_text.rstrip(", "))
			
			return "\n".join(text_parts)
			
		except Exception as e:
			logger.error(f"Failed to extract text from CSV: {e}")
			raise

	# Keep old method name for backward compatibility
	def ingest_pdf(self, client_id: Union[str, int], file_bytes: bytes, filename: str, chunk_size_words: int = 200) -> int:
		"""Legacy method - redirects to ingest_document."""
		return self.ingest_document(client_id, file_bytes, filename, chunk_size_words)

	def search_pdf_documents(self, client_id: Union[str, int], query: str, limit: int = 10) -> List[Dict[str, Any]]:
		"""Search PDF documents for a given client_id and query.
		Returns ranked results with text snippets.
		"""
		try:
			# Ensure collection exists
			self._ensure_collection_exists_by_name(TALUKA_COLLECTION_NAME)
			
			# Generate query embedding
			query_vector = self._generate_embedding(query)
			
			# Build filter for client_id
			client_id_str = str(client_id)
			search_filter = Filter(
				must=[
					FieldCondition(key='client_id', match=MatchValue(value=client_id_str)),
					FieldCondition(key='type', match=MatchValue(value='document_chunk'))  # Updated to match new type
				]
			)
			
			# Search - fetch more to ensure we get top results
			results = self.client.search(
				collection_name=TALUKA_COLLECTION_NAME,
				query_vector=query_vector,
				limit=limit * 3,  # Overfetch to ensure enough results
				query_filter=search_filter
			)
			
			if not results:
				logger.info(f"No PDF documents found for client_id={client_id_str} query={query}")
				return []
			
			# Process results - keep scores for sorting
			temp_results = []
			for res in results:
				payload = res.payload or {}
				score = float(res.score or 0.0)
				
				# Get text and truncate if too long (max 500 chars)
				full_text = payload.get('text', '')
				text_snippet = full_text[:500] + '...' if len(full_text) > 500 else full_text
				
				temp_results.append({
					'text': text_snippet.strip(),
					'_score': score  # Temporary field for sorting
				})
			
			# Sort by score (highest first)
			temp_results.sort(key=lambda x: -x['_score'])
			
			# Remove score field and return only text
			cleaned_results = [{'text': r['text']} for r in temp_results[:limit]]
			return cleaned_results
			
		except Exception as e:
			logger.error(f"Error searching PDF documents: {e}")
			return []

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
		
		# Names with moderate weight
		for name_field in ['business_name', 'service_name', 'product_name']:
			val = payload.get(name_field)
			if val:
				text_parts.extend([val] * 2)
		
		desc = payload.get('description')
		if desc:
			text_parts.append(desc)
		
		# Location fields for semantics
		for loc_field in ['city', 'state', 'country', 'pincode']:
			val = payload.get(loc_field)
			if val:
				text_parts.extend([str(val)] * 2)
		
		tags = payload.get('tags') or []
		if tags:
			lower_tags = [str(t).strip().lower() for t in tags if t]
			text_parts.extend(lower_tags * 3)
		
		return " ".join([str(p) for p in text_parts if p])

	def _normalize_query_tokens(self, query: str) -> set:
		"""Lowercase, strip punctuation, drop common stopwords, add simple de-plural forms."""
		import re
		text = query.lower()
		text = re.sub(r"[\?\!\,\.\n\t]", " ", text)
		raw_tokens = [t for t in text.split() if t]
		stop = {
			"the", "a", "an", "near", "me", "for", "to", "of", "in", "at", "find", "please",
		}
		tokens = []
		for tok in raw_tokens:
			if tok in stop:
				continue
			# simple singularization
			if tok.endswith('s') and len(tok) > 3:
				tokens.append(tok[:-1])
			tokens.append(tok)
		return set(tokens)

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
			from qdrant_client.models import GeoPoint, GeoRadius, Filter
			logger.info("geo_search_entities: lat=%s lon=%s client_id=%s query=%r", latitude, longitude, client_id, query)

			query_vector = self._generate_embedding(query)

			search_radii_meters = [2000, 5000, 10000, 20000, 50000, 100000, 200000]

			best_results: List[Dict[str, Any]] = []
			used_radius_m: Optional[int] = None

			ABS_MIN_RAW_SIM = 0.35

			for radius_m in search_radii_meters:
				# Try both string and int formats for client_id
				# Start with string format (most likely)
				client_id_str = str(client_id)
				
				# Enforce client_id and geo filters
				search_filter = Filter(
					must=[
						FieldCondition(
							key='location',
							geo_radius=GeoRadius(
								center=GeoPoint(lat=latitude, lon=longitude),
								radius=radius_m
							)
						),
						FieldCondition(key='client_id', match=MatchValue(value=client_id_str))
					]
				)

				search_results = self.client.search(
					collection_name=self.collection_name,
					query_vector=query_vector,
					limit=1000,
					query_filter=search_filter
				)
				logger.info(f"radius={radius_m}m: got {len(search_results) if search_results else 0} candidates for client_id={client_id_str}")

				if not search_results:
					logger.info(f"radius={radius_m}m: no candidates found, continuing with next radius")
					continue

				interim: List[Dict[str, Any]] = []
				# Tokenize query to compare with tags (no synonyms/hardcoded keywords)
				query_tokens = self._normalize_query_tokens(query)
				logger.debug(f"query_tokens={query_tokens}")

				for res in search_results:
					payload = res.payload or {}
					logger.debug(f"Processing result: id={res.id}, score={res.score}, payload_keys={list(payload.keys())}")
					loc = payload.get('location') or {}
					if not isinstance(loc, dict) or 'lat' not in loc or 'lon' not in loc:
						logger.debug("skip: missing_location id=%s", payload.get('business_id') or payload.get('service_id') or payload.get('product_id'))
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
					semantic_norm = max(0.0, min(1.0, raw_score))
					# Allow pass-through if tags overlap with query tokens, even if below threshold (but not too low)
					tags_lower = [str(t).strip().lower() for t in (cleaned_payload.get('tags') or [])]
					matches_tag = any(t in query_tokens for t in tags_lower)
					min_pass_score = max(0.2, SEMANTIC_SCORE_THRESHOLD * 0.6)
					
					logger.debug(f"entity: {cleaned_payload.get('business_name') or cleaned_payload.get('service_name') or cleaned_payload.get('product_name')}, score={semantic_norm:.3f}, threshold={SEMANTIC_SCORE_THRESHOLD}, matches_tag={matches_tag}, tags={tags_lower}")
					
					if not matches_tag and semantic_norm < SEMANTIC_SCORE_THRESHOLD:
						logger.debug(f"Filtered out due to semantic score {semantic_norm:.3f} < {SEMANTIC_SCORE_THRESHOLD}")
						continue
					if matches_tag and semantic_norm < min_pass_score:
						logger.debug(f"Filtered out due to semantic score {semantic_norm:.3f} < {min_pass_score:.3f} (with tag match)")
						continue

					interim.append({
						'qdrant_id': res.id,
						'raw_score': semantic_norm,
						'distance_km': round(distance_km, 6),
						'payload': cleaned_payload,
						'has_keyword_overlap': matches_tag
					})

				if not interim:
					continue

				results_scored: List[Dict[str, Any]] = []

				for item in interim:
					semantic_norm = max(0.0, min(1.0, item['raw_score']))
					distance_score = 1.0 - min((item['distance_km'] * 1000.0) / float(radius_m), 1.0)
					payload = item['payload']
					entity_type = payload.get('type')
					combined_score = SEMANTIC_WEIGHT * semantic_norm + GEO_WEIGHT * distance_score
					# Gate out items that still look weak semantically even after blending
					if semantic_norm < SEMANTIC_SCORE_THRESHOLD:
						logger.debug(f"Skipping due to final semantic threshold check: {semantic_norm:.3f} < {SEMANTIC_SCORE_THRESHOLD}")
						continue
					
					logger.debug(f"entity_type={entity_type}, payload_keys={list(payload.keys())}")
					
					if entity_type == 'business':
						domain_id = payload.get('business_id')
					elif entity_type == 'service':
						domain_id = payload.get('service_id')
					elif entity_type == 'product':
						domain_id = payload.get('product_id')
					else:
						logger.warning(f"Unknown entity_type={entity_type}, payload={payload}")
						domain_id = None

					if not domain_id:
						logger.warning(f"No domain_id found for entity_type={entity_type}, payload={payload}")
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

			# If no results found with geo constraint, try without geo to debug
			if not best_results:
				logger.warning("No results found with geo constraint, trying without geo constraint to debug")
				try:
					client_id_str = str(client_id)
					fallback_filter = Filter(must=[FieldCondition(key='client_id', match=MatchValue(value=client_id_str))])
					fallback_results = self.client.search(
						collection_name=self.collection_name,
						query_vector=query_vector,
						limit=50,
						query_filter=fallback_filter
					)
					logger.info(f"Fallback search (no geo) found {len(fallback_results) if fallback_results else 0} results for client_id={client_id_str}")
					if fallback_results:
						for res in fallback_results[:3]:  # Show first 3
							payload = res.payload or {}
							logger.info(f"Fallback result: id={res.id}, score={res.score:.3f}, name={payload.get('business_name') or payload.get('service_name') or payload.get('product_name')}, type={payload.get('type')}, client_id={payload.get('client_id')}")
				except Exception as e:
					logger.error(f"Error in fallback search: {e}")

			logger.info("geo_search_entities: selected=%s min_km=%.2f max_km=%.2f", len(best_results), min((r.get('distance_km') or 0) for r in best_results) if best_results else 0, max((r.get('distance_km') or 0) for r in best_results) if best_results else 0)
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
				limit=limit * 3,  # overfetch to allow post-filtering by threshold
				query_filter=search_filter
			)
			
			cleaned = []
			query_tokens = self._normalize_query_tokens(query)
			min_pass_score = max(0.2, SEMANTIC_SCORE_THRESHOLD * 0.6)
			for result in results:
				raw = float(result.score or 0.0)
				semantic = max(0.0, min(1.0, raw))
				payload = result.payload or {}
				tags_lower = [str(t).strip().lower() for t in (payload.get('tags') or [])]
				matches_tag = any(t in query_tokens for t in tags_lower)
				if not matches_tag and semantic < SEMANTIC_SCORE_THRESHOLD:
					continue
				if matches_tag and semantic < min_pass_score:
					continue
				cleaned.append({
					'id': result.id,
					'score': round(semantic, 4),
					'payload': payload
				})
			
			cleaned.sort(key=lambda x: -x['score'])
			return cleaned[:limit]
			
		except Exception as e:
			logger.error(f"Error searching similar entities: {e}")
			return []
