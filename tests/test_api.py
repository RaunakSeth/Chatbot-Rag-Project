"""
FastAPI endpoint tests using httpx AsyncClient + mocked pipeline.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch

from api.app import app


@pytest.mark.asyncio
@patch("api.routes.load_config")
@patch("api.routes.run_pipeline")
async def test_chat_endpoint(mock_pipeline, mock_load_config):
    from chatbot.config import ClientConfig
    from chatbot.pipeline import PipelineResponse

    mock_load_config.return_value = ClientConfig(
        client_id="test", business_name="Test", hardware_tier="A"
    )
    mock_pipeline.return_value = PipelineResponse(
        answer="We offer 24/7 support.",
        blocked=False,
        category="services",
        retrieved_count=2,
        model_used="qwen2.5:4b-instruct-q4_K_M",
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/chat",
            json={"client_id": "test", "session_id": "s1", "message": "What services do you offer?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["blocked"] is False


@pytest.mark.asyncio
@patch("api.routes.load_config")
async def test_chat_unknown_client(mock_load_config):
    mock_load_config.side_effect = FileNotFoundError("Client not found")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/chat",
            json={"client_id": "unknown", "session_id": "s1", "message": "Hello"},
        )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_health_endpoint():
    with patch("api.routes.ollama") as mock_ollama:
        mock_client = MagicMock()
        mock_client.list.return_value = MagicMock(models=[MagicMock(), MagicMock()])
        mock_ollama.Client.return_value = mock_client

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/health")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_delete_session():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.delete("/sessions/some-session-id")
    assert response.status_code == 200
    assert response.json()["exists"] is False
