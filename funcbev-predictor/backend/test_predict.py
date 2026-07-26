import sys
import os
from fastapi.testclient import TestClient

# Resolve the backend directory dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("Health check test passed!")

def test_predict_partial_payload():
    payload = {
        "responses": {
            "HO1": 5.0,
            "AT1": 5.0,
            "INNO1": 5.0
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "probability" in data
    assert "section_scores" in data
    print("Prediction with partial payload test passed! Result:", data)

def test_predict_full_payload():
    # Simulate a full payload for all 21 active features
    payload = {
        "responses": {
            "HO1": 4, "HO2": 5, "HO3": 3, "HO4": 4,
            "AT1": 5, "AT2": 2, "AT3": 4,
            "BR1": 4, "BR2": 3, "BR3": 3, "BR4": 4,
            "SN1": 2, "SN2": 2, "SN4": 4,
            "PBC1": 4, "PBC2": 3, "PBC3": 2, "PBC4": 3,
            "INNO1": 4, "INNO2": 3, "INNO4": 4
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "probability" in data
    assert "section_scores" in data
    print("Prediction with full payload test passed! Result:", data)

if __name__ == "__main__":
    print("Running integration tests...")
    try:
        test_health()
        test_predict_partial_payload()
        test_predict_full_payload()
        print("ALL TESTS PASSED SUCCESSFULLY!")
    except Exception as e:
        import traceback
        print("TEST RUN FAILED!")
        traceback.print_exc()
        sys.exit(1)
