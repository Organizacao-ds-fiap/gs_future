from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from routers import predicts
from utils.models_loader import load_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()

    yield

app = FastAPI(
    title="LLM Matchmaker API",
    docs_url="/docs",  # URL para disponibilização do Swagger UI
    lifespan=lifespan
)

# Libera o CORS da API para requisições via http
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

app.include_router(predicts.router)