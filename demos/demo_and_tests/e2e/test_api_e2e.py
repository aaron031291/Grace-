"""
Basic E2E Test for Grace API Health and Auth
"""

import requests

import pytest


@pytest.mark.e2e
def test_health():
    resp = requests.get("http://localhost:8000/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@pytest.mark.e2e
def test_auth_token():
    resp = requests.post(
        "http://localhost:8000/api/auth/token",
        json={"username": "admin", "password": "admin"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data


@pytest.mark.e2e
def test_auth_refresh():
    token_resp = requests.post(
        "http://localhost:8000/api/auth/token",
        json={"username": "admin", "password": "admin"},
    )
    refresh_token = token_resp.json()["refresh_token"]
    resp = requests.post(
        "http://localhost:8000/api/auth/refresh", json={"refresh_token": refresh_token}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
