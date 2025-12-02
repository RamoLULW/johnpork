from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from camion import run_simulation_return_history
from fastapi.responses import JSONResponse
import numpy as np

app = FastAPI(title="AgentiPy Simulation")

class SimRequest(BaseModel):
    N: int = 12
    T: int = 4
    capacity: int = 10
    p_good: float = 0.5
    seed: int = 42
    max_ticks: int = 500
    use_qlearning: bool = False

def fix_numpy(obj):
    if isinstance(obj, dict):
        return {k: fix_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fix_numpy(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

@app.get("/")
def root():
    return {"status": "running", "endpoints": ["/simulated"]}

@app.post("/simulated")
def json_camion(req: SimRequest):
    history = run_simulation_return_history(
        req.N,
        req.T,
        req.capacity,
        req.p_good,
        req.seed,
        req.max_ticks,
        req.use_qlearning,
    )
    history = fix_numpy(history)
    plants = history.get("plants", [])
    resp = {
        "plants": plants,
        "pos": history["pos"],
        "load": history["load"],
        "tick": history["tick"],
    }
    return JSONResponse(content=resp)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
