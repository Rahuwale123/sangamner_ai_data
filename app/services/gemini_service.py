import logging
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from app.core.config import GEMINI_API_KEY, GEMINI_API_URL

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google Gemini AI API"""
    
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.api_url = GEMINI_API_URL
        self.timeout = 5.0  # 5 seconds timeout for fast responses
        
    async def generate_search_summary(self, search_results: List[Dict[str, Any]], query: str, location: str) -> str:
        """
        Generate AI summary of search results using Gemini API
        
        Args:
            search_results: List of search results with business/service/product data
            query: Original search query
            location: Location context (city/area)
            
        Returns:
            AI-generated summary string
        """
        try:
            if not search_results:
                return "No results found for your search query."
            
            # Prepare context for AI
            context = self._prepare_search_context(search_results, query, location)
            
            # Generate AI response
            ai_response = await self._call_gemini_api(context)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error generating AI summary: {e}")
            return "AI summary is currently unavailable. Please try again later."
    
    def _prepare_search_context(self, search_results: List[Dict[str, Any]], query: str, location: str) -> str:
        """Prepare search context for AI processing"""
        
        # Extract key information from results
        businesses = []
        services = []
        products = []
        
        for result in search_results[:10]:  # Limit to top 10 results for context
            payload = result.get('payload', {})
            entity_type = payload.get('type', '')
            distance = result.get('distance_km', 0)
            
            entity_info = {
                'name': self._get_entity_name(payload),
                'distance': f"{distance:.1f} km",
                'description': payload.get('description', ''),
                'tags': payload.get('tags', [])
            }
            
            if entity_type == 'business':
                businesses.append(entity_info)
            elif entity_type == 'service':
                services.append(entity_info)
            elif entity_type == 'product':
                products.append(entity_info)
        
        # Build context string
        context_parts = [
            f"Search Query: '{query}' in {location}",
            f"Found {len(search_results)} results:",
            ""
        ]
        
        if businesses:
            context_parts.append("BUSINESSES:")
            for i, biz in enumerate(businesses, 1):
                context_parts.append(f"{i}. {biz['name']} ({biz['distance']})")
                if biz['description']:
                    context_parts.append(f"   Description: {biz['description']}")
                if biz['tags']:
                    context_parts.append(f"   Tags: {', '.join(biz['tags'])}")
            context_parts.append("")
        
        if services:
            context_parts.append("SERVICES:")
            for i, svc in enumerate(services, 1):
                context_parts.append(f"{i}. {svc['name']} ({svc['distance']})")
                if svc['description']:
                    context_parts.append(f"   Description: {svc['description']}")
                if svc['tags']:
                    context_parts.append(f"   Tags: {', '.join(svc['tags'])}")
            context_parts.append("")
        
        if products:
            context_parts.append("PRODUCTS:")
            for i, prod in enumerate(products, 1):
                context_parts.append(f"{i}. {prod['name']} ({prod['distance']})")
                if prod['description']:
                    context_parts.append(f"   Description: {prod['description']}")
                if prod['tags']:
                    context_parts.append(f"   Tags: {', '.join(prod['tags'])}")
            context_parts.append("")
        
        context_parts.extend([
            "Please provide a brief, helpful summary of these search results.",
            "Focus on the most relevant options and their proximity to the user.",
            "Keep the response concise (2-3 sentences) and user-friendly.",
            "Highlight any particularly interesting or unique options."
        ])
        
        return "\n".join(context_parts)
    
    def _get_entity_name(self, payload: Dict[str, Any]) -> str:
        """Extract entity name from payload"""
        return (
            payload.get('business_name') or 
            payload.get('service_name') or 
            payload.get('product_name') or 
            'Unknown'
        )
    
    async def _call_gemini_api(self, context: str) -> str:
        """Call Gemini API with the prepared context"""
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": context
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 200,  # Keep response concise
                "temperature": 0.7,
                "topP": 0.8,
                "topK": 40
            }
        }
        
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': self.api_key
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract generated text from response
                        if 'candidates' in result and len(result['candidates']) > 0:
                            candidate = result['candidates'][0]
                            if 'content' in candidate and 'parts' in candidate['content']:
                                parts = candidate['content']['parts']
                                if len(parts) > 0 and 'text' in parts[0]:
                                    return parts[0]['text'].strip()
                        
                        return "AI response generated successfully."
                    else:
                        error_text = await response.text()
                        logger.error(f"Gemini API error {response.status}: {error_text}")
                        return "AI summary is currently unavailable."
                        
        except asyncio.TimeoutError:
            logger.error("Gemini API timeout")
            return "AI summary is currently unavailable due to timeout."
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "AI summary is currently unavailable."
    
    async def test_connection(self) -> bool:
        """Test Gemini API connection"""
        try:
            test_payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": "Say 'Hello' in one word"
                            }
                        ]
                    }
                ]
            }
            
            headers = {
                'Content-Type': 'application/json',
                'X-goog-api-key': self.api_key
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3.0)) as session:
                async with session.post(self.api_url, headers=headers, json=test_payload) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Gemini connection test failed: {e}")
            return False