
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, get_db
from main import app
import pytest

# Setup in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite://"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module")
def client():
    # Create tables
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c
    # Drop tables
    Base.metadata.drop_all(bind=engine)

def test_create_tier3_result(client):
    response = client.post(
        "/api/tier3/",
        json={
            "smiles": "C",
            "result": {"energy": -50.5, "gap": 2.3}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 201
    assert data["data"]["smiles"] == "C"
    assert data["data"]["result"]["energy"] == -50.5

def test_get_tier3_result(client):
    response = client.get("/api/tier3/C")
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["smiles"] == "C"
    assert data["data"]["result"]["gap"] == 2.3

def test_get_tier3_result_not_found(client):
    response = client.get("/api/tier3/NONEXISTENT")
    assert response.status_code == 404

def test_create_duplicate_tier3_result(client):
    response = client.post(
        "/api/tier3/",
        json={
            "smiles": "C",
            "result": {"energy": -100.0}
        }
    )
    assert response.status_code == 400

def test_update_tier3_result(client):
    response = client.put(
        "/api/tier3/C",
        json={
            "result": {"energy": -60.0, "gap": 2.5}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["result"]["energy"] == -60.0

def test_delete_tier3_result(client):
    response = client.delete("/api/tier3/C")
    assert response.status_code == 200
    
    # Verify it's gone
    response = client.get("/api/tier3/C")
    assert response.status_code == 404
