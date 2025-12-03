from fastapi import FastAPI
from app.api.routes_simulation import router as simulation_router

app = FastAPI(title="Tractor Multi-Agent Simulation")

app.include_router(simulation_router, prefix="/api")
