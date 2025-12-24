"""
Tests for API Endpoints.

Tests the refactored endpoint modules.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def mock_registry():
    """Create a mock service registry."""
    with patch('app.api.endpoints.chat.get_registry') as chat_mock, \
         patch('app.api.endpoints.knowledge.get_registry') as knowledge_mock, \
         patch('app.api.endpoints.agent.get_registry') as agent_mock, \
         patch('app.api.endpoints.mcp.get_registry') as mcp_mock, \
         patch('app.api.endpoints.system.get_registry') as system_mock, \
         patch('app.api.endpoints.config.get_registry') as config_mock:

        # Create mock services
        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value="RAG response")
        mock_rag.get_relevant_documents = MagicMock(return_value=[])
        mock_rag.async_add_knowledge = AsyncMock(return_value={"success": True, "num_nodes": 5})
        mock_rag.async_index_directory = AsyncMock(return_value={"success": True, "num_documents": 10})

        mock_chat = MagicMock()
        mock_chat.chat = MagicMock(return_value="Chat response")

        mock_llm = MagicMock()
        mock_llm.provider = "ollama"
        mock_llm.model = "test-model"
        mock_llm.base_url = "http://localhost:11434"
        mock_llm.temperature = 0.7
        mock_llm.max_tokens = 2048
        mock_llm.supports_tool_calling = True
        mock_llm.get_info = MagicMock(return_value={"provider": "ollama", "model": "test-model"})

        mock_vector_store = MagicMock()
        mock_vector_store.get_collection_stats = MagicMock(return_value={"count": 100})
        mock_vector_store.delete_collection = MagicMock(return_value=True)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value={
            "response": "Agent response",
            "iterations": 2,
            "tool_calls": [],
        })

        mock_tool_registry = MagicMock()
        mock_tool_registry.get_all = MagicMock(return_value=[])
        mock_tool_registry.get = MagicMock(return_value=None)

        mock_memory_manager = MagicMock()
        mock_memory_manager.get_memory = MagicMock(return_value=None)

        mock_upload_service = MagicMock()

        # Create registry mock
        registry = MagicMock()
        registry.get = MagicMock(side_effect=lambda name: {
            "rag": mock_rag,
            "chat": mock_chat,
            "llm": mock_llm,
            "vector_store": mock_vector_store,
            "agent": mock_agent,
            "langgraph_agent": None,
            "tool_registry": mock_tool_registry,
            "memory_manager": mock_memory_manager,
            "upload_service": mock_upload_service,
        }.get(name))

        # Apply to all patches
        for mock in [chat_mock, knowledge_mock, agent_mock, mcp_mock, system_mock, config_mock]:
            mock.return_value = registry

        yield registry


@pytest.fixture
def client(mock_registry):
    """Create a test client."""
    from app.main import app
    return TestClient(app)


class TestChatEndpoints:
    """Test chat endpoints."""

    def test_send_message_with_rag(self, client, mock_registry):
        """Test sending message with RAG."""
        response = client.post(
            "/api/chat/message",
            json={
                "message": "What is Python?",
                "use_rag": True,
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    def test_send_message_without_rag(self, client, mock_registry):
        """Test sending message without RAG."""
        response = client.post(
            "/api/chat/message",
            json={
                "message": "Hello",
                "use_rag": False,
                "stream": False,
            },
        )
        assert response.status_code == 200


class TestKnowledgeEndpoints:
    """Test knowledge base endpoints."""

    def test_add_text_knowledge(self, client, mock_registry):
        """Test adding text knowledge."""
        response = client.post(
            "/api/knowledge/add-text",
            json={
                "text": "This is a test document.",
                "title": "Test",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_search_knowledge(self, client, mock_registry):
        """Test searching knowledge base."""
        mock_registry.get("rag").get_relevant_documents.return_value = [
            {"content": "Result 1", "score": 0.9},
        ]
        response = client.post(
            "/api/knowledge/search",
            json={
                "query": "test query",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_get_stats(self, client, mock_registry):
        """Test getting knowledge base stats."""
        with patch('app.services.knowledge_base_service.get_knowledge_base_service') as mock_kb:
            mock_kb.return_value.get_stats.return_value = {"total_files": 10}

            response = client.get("/api/knowledge/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_clear_knowledge_base(self, client, mock_registry):
        """Test clearing knowledge base."""
        response = client.delete("/api/knowledge/clear")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestSystemEndpoints:
    """Test system endpoints."""

    def test_health_check(self, client, mock_registry):
        """Test health check endpoint."""
        response = client.get("/api/system/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_get_system_info(self, client, mock_registry):
        """Test getting system info."""
        response = client.get("/api/system/info")
        assert response.status_code == 200
        data = response.json()
        assert "app_name" in data
        assert "version" in data

    def test_get_config(self, client, mock_registry):
        """Test getting config."""
        response = client.get("/api/system/config")
        assert response.status_code == 200
        data = response.json()
        assert "llm_provider" in data


class TestAgentEndpoints:
    """Test agent endpoints."""

    @patch('app.api.endpoints.agent.settings')
    def test_agent_chat(self, mock_settings, client, mock_registry):
        """Test agent chat endpoint."""
        mock_settings.AGENT_ENABLED = True

        response = client.post(
            "/api/agent/chat",
            json={
                "message": "Search for Python documentation",
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    @patch('app.api.endpoints.agent.settings')
    def test_agent_disabled(self, mock_settings, client, mock_registry):
        """Test agent when disabled."""
        mock_settings.AGENT_ENABLED = False

        response = client.post(
            "/api/agent/chat",
            json={"message": "Test", "stream": False},
        )
        assert response.status_code == 403

    @patch('app.api.endpoints.agent.settings')
    @patch('app.api.endpoints.agent.LANGGRAPH_AVAILABLE', True)
    def test_get_capabilities(self, mock_settings, client, mock_registry):
        """Test getting agent capabilities."""
        mock_settings.AGENT_MAX_ITERATIONS = 10

        response = client.get("/api/agent/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert "react_agent" in data
        assert "max_iterations" in data


class TestMCPEndpoints:
    """Test MCP tool endpoints."""

    def test_list_tools(self, client, mock_registry):
        """Test listing MCP tools."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.parameters = []

        mock_registry.get("tool_registry").get_all.return_value = [mock_tool]

        response = client.get("/api/mcp/tools")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_tool_not_found(self, client, mock_registry):
        """Test getting non-existent tool."""
        mock_registry.get("tool_registry").get.return_value = None

        response = client.get("/api/mcp/tools/nonexistent")
        assert response.status_code == 404


class TestConfigEndpoints:
    """Test configuration endpoints."""

    def test_get_current_config(self, client, mock_registry):
        """Test getting current config."""
        response = client.get("/api/config/current")
        assert response.status_code == 200
        data = response.json()
        assert "provider" in data
        assert "model" in data

    def test_get_providers(self, client, mock_registry):
        """Test getting available providers."""
        with patch('app.api.endpoints.config.LLMManager') as mock_llm:
            mock_llm.get_available_providers.return_value = [
                {
                    "id": "ollama",
                    "name": "Ollama",
                    "models": ["llama3"],
                    "requires_api_key": False,
                    "base_url": "http://localhost:11434",
                },
            ]

            response = client.get("/api/config/providers")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_validate_config(self, client, mock_registry):
        """Test validating config."""
        with patch('app.api.endpoints.config.LLMManager') as mock_llm:
            mock_llm.validate_provider_config.return_value = {
                "valid": True,
                "provider": "ollama",
                "name": "Ollama",
            }

            response = client.post(
                "/api/config/validate",
                json={"provider": "ollama"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True


class TestSchemaValidation:
    """Test request/response schema validation."""

    def test_chat_request_validation(self, client, mock_registry):
        """Test chat request validation."""
        # Missing required field
        response = client.post(
            "/api/chat/message",
            json={"use_rag": True},  # Missing 'message'
        )
        assert response.status_code == 422  # Validation error

    def test_search_request_validation(self, client, mock_registry):
        """Test search request validation."""
        # Invalid top_k type
        response = client.post(
            "/api/knowledge/search",
            json={"query": "test", "top_k": "invalid"},
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])