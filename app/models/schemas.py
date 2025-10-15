from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum

class EntityType(str, Enum):
	BUSINESS = "business"
	SERVICE = "service"
	PRODUCT = "product"

class Location(BaseModel):
	lat: float = Field(..., description="Latitude coordinate")
	lon: float = Field(..., description="Longitude coordinate")

class BusinessPayload(BaseModel):
	type: EntityType = EntityType.BUSINESS
	business_id: str = Field(..., description="Unique business identifier")
	business_name: str = Field(..., description="Name of the business")
	group_id: Optional[int] = Field(None, description="Category ID for grouping")
	location: Location = Field(..., description="Business coordinates")
	address: Optional[str] = Field(None, description="Business address")
	phone: Optional[str] = Field(None, description="Contact phone number")
	description: Optional[str] = Field(None, description="Business description")
	tags: Optional[List[str]] = Field(default_factory=list, description="Business tags")
	client_id: Optional[str] = Field(None, description="Client identifier")
	user_id: Optional[str] = Field(None, description="User identifier")
	business_type: Optional[str] = Field(None, description="Type of business")
	status: Optional[str] = Field(None, description="Business status")
	pincode: Optional[str] = Field(None, description="Postal code")
	city: Optional[str] = Field(None, description="City")
	state: Optional[str] = Field(None, description="State")
	country: Optional[str] = Field(None, description="Country")
	email: Optional[str] = Field(None, description="Email")
	website: Optional[str] = Field(None, description="Website")

class ServicePayload(BaseModel):
	type: EntityType = EntityType.SERVICE
	service_id: str = Field(..., description="Unique service identifier")
	business_id: Optional[str] = Field(None, description="Parent business identifier (optional)")
	service_name: str = Field(..., description="Name of the service")
	group_id: Optional[int] = Field(None, description="Category ID for grouping")
	location: Location = Field(..., description="Service coordinates (mandatory)")
	price: Optional[str] = Field(None, description="Service price")
	description: Optional[str] = Field(None, description="Service description")
	tags: Optional[List[str]] = Field(default_factory=list, description="Service tags")
	client_id: Optional[str] = Field(None, description="Client identifier")
	user_id: Optional[str] = Field(None, description="User identifier")

class ProductPayload(BaseModel):
	type: EntityType = EntityType.PRODUCT
	product_id: str = Field(..., description="Unique product identifier")
	service_id: Optional[str] = Field(None, description="Parent service identifier")
	business_id: str = Field(..., description="Parent business identifier")
	product_name: str = Field(..., description="Name of the product")
	group_id: Optional[int] = Field(None, description="Category ID for grouping")
	location: Location = Field(..., description="Product coordinates (mandatory)")
	price: Optional[str] = Field(None, description="Product price")
	description: Optional[str] = Field(None, description="Product description")
	tags: Optional[List[str]] = Field(default_factory=list, description="Product tags")
	client_id: Optional[str] = Field(None, description="Client identifier")

class SaveBusinessRequest(BaseModel):
	type: EntityType = EntityType.BUSINESS
	business_id: str
	group_id: int
	business_name: str
	location: Location
	address: Optional[str] = None
	phone: Optional[str] = None
	description: Optional[str] = None
	tags: Optional[List[str]] = None

class SaveServiceRequest(BaseModel):
	type: EntityType = EntityType.SERVICE
	service_id: str
	business_id: Optional[str] = None
	group_id: int
	service_name: str
	location: Location
	price: Optional[str] = None
	description: Optional[str] = None
	tags: Optional[List[str]] = None

class SaveProductRequest(BaseModel):
	type: EntityType = EntityType.PRODUCT
	product_id: str
	service_id: Optional[str] = None
	business_id: str
	group_id: int
	product_name: str
	location: Location
	price: Optional[str] = None
	description: Optional[str] = None
	tags: Optional[List[str]] = None

SaveRequest = Union[SaveBusinessRequest, SaveServiceRequest, SaveProductRequest]

class UpdateRequest(BaseModel):
	business_name: Optional[str] = None
	service_name: Optional[str] = None
	product_name: Optional[str] = None
	address: Optional[str] = None
	phone: Optional[str] = None
	location: Optional[Location] = None
	price: Optional[str] = None
	description: Optional[str] = None
	tags: Optional[List[str]] = None

class SaveResponse(BaseModel):
	status: str = "success"
	id: str

class UpdateResponse(BaseModel):
	status: str = "updated"
	id: str
#
class GeoSearchRequest(BaseModel):
    latitude: float = Field(..., description="User's latitude coordinate")
    longitude: float = Field(..., description="User's longitude coordinate")
    client_id: str = Field(..., description="Client ID to filter entities")
    query: str = Field(..., description="Search query text (e.g., 'tea', 'breakfast', 'restaurant')")

class ErrorResponse(BaseModel):
	status: str = "error"
	message: str
	details: Optional[Dict[str, Any]] = None

class Business(BaseModel):
	business_id: str
	name: str
	description: Optional[str] = None
	business_type: Optional[str] = None
	user_id: Optional[str] = None
	client_id: str
	status: Optional[str] = "active"
	pincode: Optional[str] = None
	city: Optional[str] = None
	state: Optional[str] = None
	country: Optional[str] = None
	phone: Optional[str] = None
	email: Optional[str] = None
	website: Optional[str] = None
	lat: float
	long: float
	tags: List[str] = []

class Service(BaseModel):
	service_id: str
	service_name: str
	description: Optional[str] = None
	business_id: Optional[str] = None
	client_id: str
	user_id: Optional[str] = None
	lat: float
	long: float
	tags: List[str] = []

class Product(BaseModel):
	product_id: str
	product_name: str
	description: Optional[str] = None
	business_id: Optional[str] = None
	client_id: str
	lat: float
	long: float
	tags: List[str] = []
