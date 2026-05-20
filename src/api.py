from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Identity Experts MoE API")

class PredictRequest(BaseModel):
    hierarchical_id: str  # formato a.b.c.d.e.f.g.h

class PredictResponse(BaseModel):
    expert_id: str
    prediction: dict
    routing_time_ms: float

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": False}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # TODO: integrar modelo real cuando esté entrenado
    return PredictResponse(
        expert_id=request.hierarchical_id,
        prediction={"value": 0.0, "confidence": 0.0},
        routing_time_ms=0.0
    )