"""FastAPI application with comprehensive middleware stack."""
from contextlib import asynccontextmanager
from typing import Callable
import time
import uuid
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from fastapi import Depends, HTTPException 
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional

from pe_orgair.config.settings import settings
from pe_orgair.api.routes.v1 import router as v1_router
from pe_orgair.api.routes.v2 import router as v2_router
# from pe_orgair.api.routes import health
# from pe_orgair.observability.setup import setup_tracing, setup_logging

logger = structlog.get_logger()

import redis
import httpx
import json 
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
CACHE_TTL = 30  # Cache time-to-live in seconds

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    """Application lifespan with startup/shutdown."""
    # Startup
    logger.info("starting_application",
                app_name=settings.APP_NAME,
                version=settings.APP_VERSION,
                env=settings.APP_ENV)
    
    # Initialize connections, caches, etc.
    # await initialize_redis()
  
    
    yield
    
    # Shutdown
    logger.info("shutting_down_application")

def create_app() -> FastAPI:
    """Application factory."""
    # setup_logging()
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.DEBUG else [],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request correlation middleware
    @app.middleware("http")
    async def add_correlation_id(request: Request, call_next: Callable) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
        
        start_time = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Process-Time"] = f"{duration:.4f}"
        
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
        )
        
        return response
    
    @app.get("/", tags=["Health"])
    async def root():
        return {"message": f"{settings.APP_NAME} is running.", "version": settings.APP_VERSION} 


    from pydantic import BaseModel
    class User(BaseModel):
        name: str
        age: int   

    @app.post("/greet")
    def greet_user(user: User):
        if user.age < 18:
            return {"message": f"Hello, {user.name}! You're a minor."}
        else:
            return {"message": f"Hello, {user.name}! Welcome to adulthood."}



    @app.get("/entries/no-cache", tags=["External API"])
    def get_entries_no_cache():
        """GET: Fetch from external API WITHOUT caching."""
        start = time.perf_counter()
        
        with httpx.Client() as client:
            response = client.get("https://api.nationalize.io/?name=nathaniel")
            data = response.json()
        
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("entries_no_cache", duration_ms=round(duration_ms, 2))
        
        return data

    @app.get("/entries", tags=["External API"])
    def get_entries_with_cache():
        """GET: Fetch from external API WITH Redis caching."""
        cache_key = "entries"
        start = time.perf_counter()
        
        # Check cache first
        cached_data = redis_client.get(cache_key)
        
        if cached_data:
            # CACHE HIT
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info("entries_cache_hit", duration_ms=round(duration_ms, 2))
            return json.loads(cached_data)
        
        # CACHE MISS - Fetch from external API
        logger.info("entries_cache_miss")
        
        with httpx.Client() as client:
            response = client.get("https://api.nationalize.io/?name=nathaniel")
            data = response.json()
        
        # Store in Redis
        redis_client.set(cache_key, json.dumps(data), ex=CACHE_TTL)
        
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("entries_fetched_from_api", duration_ms=round(duration_ms, 2))
        
        return data

    @app.delete("/entries/cache", tags=["External API"])
    def clear_entries_cache():
        """DELETE: Clear the entries cache."""
        deleted = redis_client.delete("entries")
        return {"message": "Cache cleared", "keys_deleted": deleted}

    from pe_orgair.models.item import Item 
    items_db = []
    @app.get("/items", response_model=list[Item])
    def get_items():
        """GET: Retrieve all items"""
        return items_db

    @app.post("/items", response_model=Item)
    def create_item(item: Item):
        """POST: Add a new item"""
        items_db.append(item)
        return item

    @app.put("/items/{item_id}", response_model=Item)
    def update_item(item_id: int, updated_item: Item):
        """PUT: Update an existing item"""
        for i, item in enumerate(items_db):
            if item.id == item_id:
                items_db[i] = updated_item
                return updated_item
        return {"error": "Item not found"}

    @app.delete("/items/{item_id}")
    def delete_item(item_id: int):
        """DELETE: Remove an item"""
        global items_db
        items_db = [item for item in items_db if item.id != item_id]
        return {"message": f"Item with ID {item_id} has been deleted"}

    
    # Global error handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("unhandled_exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "type": "https://api.pe-orgair.example.com/errors/internal",
                "title": "Internal Server Error",
                "status": 500,
                "detail": str(exc) if settings.DEBUG else "An internal error occurred",
            },
        )
    
    # Tracing
    # setup_tracing(app)
    
    # Routes
    # app.include_router(health.router, tags=["Health"])
    app.include_router(v1_router, prefix=settings.API_V1_PREFIX)
    app.include_router(v2_router, prefix=settings.API_V2_PREFIX)
    
    return app

app = create_app()