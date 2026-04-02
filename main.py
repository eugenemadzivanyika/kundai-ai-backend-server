import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import bkt, developmentPlan, devplan_content_generation, health, ocr, asag, agents, content, resources, assessment_generation
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()  # This must run before routers are included

# List the origins you trust
origins = [
    "http://localhost:5173",  # Your React Frontend
    "http://127.0.0.1:5173",
]

app = FastAPI(
    title="KundAI AI Services Backend",
    version="0.1.0",
    description="FastAPI service for AI/ML endpoints.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allow Content-Type, Authorization, etc.
)

app.include_router(health.router)
app.include_router(ocr.router)
app.include_router(asag.router)
app.include_router(bkt.router)
app.include_router(agents.router)
app.include_router(developmentPlan.router)
app.include_router(content.router)
app.include_router(resources.router)
app.include_router(devplan_content_generation.router)
app.include_router(assessment_generation.router)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
