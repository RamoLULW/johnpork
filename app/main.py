from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes_simulation import router as simulation_router

app = FastAPI(title="Tractor Multi-Agent Simulation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulation_router, prefix="/api")
