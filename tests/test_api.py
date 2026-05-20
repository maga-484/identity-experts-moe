from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}

def test_predict_structure():
    response = client.post("/predict", json={"hierarchical_id": "a.b.c.d"})
    assert response.status_code == 200
    data = response.json()
    assert "expert_id" in data
    assert "prediction" in data
    assert "routing_time_ms" in data