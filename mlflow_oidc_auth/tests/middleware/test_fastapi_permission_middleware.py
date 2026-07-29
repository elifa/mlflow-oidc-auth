"""
Tests for FastAPI Permission Middleware.

This module tests the OIDC-aware permission middleware for FastAPI-native routes
(gateway invocations, OTel trace ingestion, assistant, job API) that bypass Flask.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import PlainTextResponse

# ---------------------------------------------------------------------------
# Unit tests: _extract_gateway_endpoint_name
# ---------------------------------------------------------------------------


class TestExtractGatewayEndpointName:
    """Test endpoint name extraction from gateway URL patterns."""

    def _extract(self, path, body=None):
        from mlflow_oidc_auth.middleware.fastapi_permission_middleware import (
            _extract_gateway_endpoint_name,
        )

        return _extract_gateway_endpoint_name(path, body)

    def test_invocations_route(self):
        """Test /gateway/{endpoint_name}/mlflow/invocations pattern."""
        assert self._extract("/gateway/my-endpoint/mlflow/invocations") == "my-endpoint"

    def test_invocations_route_complex_name(self):
        """Test invocations route with complex endpoint name."""
        assert self._extract("/gateway/gpt-4-turbo/mlflow/invocations") == "gpt-4-turbo"

    def test_invocations_route_no_match(self):
        """Test invocations pattern with wrong suffix."""
        assert self._extract("/gateway/my-endpoint/mlflow/other") is None

    def test_proxy_route_with_trailing_path(self):
        """Test /gateway/proxy/{endpoint_name}/{path:path} pattern."""
        assert self._extract("/gateway/proxy/my-endpoint/models") == "my-endpoint"

    def test_proxy_route_with_multi_segment_path(self):
        """Test proxy route with a multi-segment trailing path."""
        assert self._extract("/gateway/proxy/my-endpoint/v1/chat/completions") == "my-endpoint"

    def test_proxy_route_without_trailing_path(self):
        """Test proxy route with no trailing path."""
        assert self._extract("/gateway/proxy/my-endpoint") == "my-endpoint"

    def test_chat_completions_mlflow(self):
        """Test MLflow chat completions passthrough."""
        result = self._extract("/gateway/mlflow/v1/chat/completions", {"model": "my-model"})
        assert result == "my-model"

    def test_chat_completions_openai(self):
        """Test OpenAI chat completions passthrough."""
        result = self._extract("/gateway/openai/v1/chat/completions", {"model": "gpt-4"})
        assert result == "gpt-4"

    def test_embeddings_openai(self):
        """Test OpenAI embeddings passthrough."""
        result = self._extract("/gateway/openai/v1/embeddings", {"model": "text-embedding-ada"})
        assert result == "text-embedding-ada"

    def test_responses_openai(self):
        """Test OpenAI responses passthrough."""
        result = self._extract("/gateway/openai/v1/responses", {"model": "gpt-4o"})
        assert result == "gpt-4o"

    def test_responses_compact_openai(self):
        """Test OpenAI compact responses passthrough."""
        result = self._extract("/gateway/openai/v1/responses/compact", {"model": "ep"})
        assert result == "ep"

    def test_anthropic_messages(self):
        """Test Anthropic messages passthrough."""
        result = self._extract("/gateway/anthropic/v1/messages", {"model": "claude-3"})
        assert result == "claude-3"

    def test_passthrough_no_body(self):
        """Test passthrough route with no body returns None."""
        assert self._extract("/gateway/mlflow/v1/chat/completions", None) is None

    def test_passthrough_no_model_in_body(self):
        """Test passthrough route with body missing 'model' key."""
        assert self._extract("/gateway/mlflow/v1/chat/completions", {"prompt": "hi"}) is None

    def test_gemini_generate_content(self):
        """Test Gemini generateContent pattern."""
        result = self._extract("/gateway/gemini/v1beta/models/gemini-pro:generateContent")
        assert result == "gemini-pro"

    def test_gemini_stream_generate_content(self):
        """Test Gemini streamGenerateContent pattern."""
        result = self._extract("/gateway/gemini/v1beta/models/gemini-ultra:streamGenerateContent")
        assert result == "gemini-ultra"

    def test_gemini_no_match(self):
        """Test Gemini pattern with wrong action."""
        assert self._extract("/gateway/gemini/v1beta/models/foo:otherAction") is None

    def test_unrelated_path(self):
        """Test unrelated paths return None."""
        assert self._extract("/api/experiments") is None
        assert self._extract("/v1/traces") is None


# ---------------------------------------------------------------------------
# Unit tests: _find_fastapi_validator
# ---------------------------------------------------------------------------


class TestFindFastapiValidator:
    """Test route-to-validator dispatcher."""

    def _find(self, path):
        from mlflow_oidc_auth.middleware.fastapi_permission_middleware import (
            _find_fastapi_validator,
        )

        return _find_fastapi_validator(path)

    def test_gateway_route_returns_validator(self):
        """Test gateway routes return a validator."""
        assert self._find("/gateway/my-endpoint/mlflow/invocations") is not None

    def test_otel_route_returns_validator(self):
        """Test OTel routes return a validator."""
        assert self._find("/v1/traces") is not None
        assert self._find("/v1/traces/something") is not None

    def test_jobs_route_returns_validator(self):
        """Test job API routes return a validator."""
        assert self._find("/ajax-api/3.0/jobs") is not None
        assert self._find("/ajax-api/3.0/jobs/some-job") is not None

    def test_assistant_route_returns_validator(self):
        """Test assistant routes return a validator."""
        assert self._find("/ajax-api/3.0/mlflow/assistant") is not None
        assert self._find("/ajax-api/3.0/mlflow/assistant/chat") is not None

    def test_flask_route_returns_none(self):
        """Test Flask-handled routes return None (pass-through)."""
        assert self._find("/api/2.0/mlflow/experiments/list") is None
        assert self._find("/health") is None
        assert self._find("/oidc/ui") is None

    def test_root_returns_none(self):
        """Test root path returns None."""
        assert self._find("/") is None


# ---------------------------------------------------------------------------
# Unit tests: gateway validator logic
# ---------------------------------------------------------------------------


class TestGatewayValidator:
    """Test gateway validator permission checking."""

    def _get_gateway_validator(self, path):
        from mlflow_oidc_auth.middleware.fastapi_permission_middleware import (
            _get_gateway_validator,
        )

        return _get_gateway_validator(path)

    @pytest.mark.asyncio
    @patch("mlflow_oidc_auth.middleware.fastapi_permission_middleware.can_use_gateway_endpoint")
    async def test_invocations_allowed(self, mock_can_use):
        """Test gateway invocation with USE permission succeeds."""
        mock_can_use.return_value = True
        validator = self._get_gateway_validator("/gateway/my-endpoint/mlflow/invocations")
        request = MagicMock(spec=Request)
        result = await validator("user@example.com", request)
        assert result is True
        mock_can_use.assert_called_once_with("my-endpoint", "user@example.com")

    @pytest.mark.asyncio
    @patch("mlflow_oidc_auth.middleware.fastapi_permission_middleware.can_use_gateway_endpoint")
    async def test_invocations_denied(self, mock_can_use):
        """Test gateway invocation without USE permission fails."""
        mock_can_use.return_value = False
        validator = self._get_gateway_validator("/gateway/my-endpoint/mlflow/invocations")
        request = MagicMock(spec=Request)
        result = await validator("user@example.com", request)
        assert result is False

    @pytest.mark.asyncio
    @patch("mlflow_oidc_auth.middleware.fastapi_permission_middleware.can_use_gateway_endpoint")
    async def test_passthrough_reads_body(self, mock_can_use):
        """Test passthrough route reads body to extract model name."""
        mock_can_use.return_value = True
        validator = self._get_gateway_validator("/gateway/openai/v1/chat/completions")
        request = MagicMock(spec=Request)
        request.json = AsyncMock(return_value={"model": "gpt-4"})
        request.state = MagicMock()

        result = await validator("user@example.com", request)
        assert result is True
        mock_can_use.assert_called_once_with("gpt-4", "user@example.com")

    @pytest.mark.asyncio
    async def test_passthrough_bad_json(self):
        """Test passthrough route with unparseable body returns False."""
        validator = self._get_gateway_validator("/gateway/openai/v1/chat/completions")
        request = MagicMock(spec=Request)
        request.json = AsyncMock(side_effect=ValueError("bad json"))
        request.state = MagicMock()

        result = await validator("user@example.com", request)
        assert result is False

    @pytest.mark.asyncio
    @patch("mlflow_oidc_auth.middleware.fastapi_permission_middleware.can_use_gateway_endpoint")
    async def test_proxy_get_does_not_read_body(self, mock_can_use):
        """Test GET proxy route resolves the endpoint from the URL without reading the body."""
        mock_can_use.return_value = True
        validator = self._get_gateway_validator("/gateway/proxy/my-endpoint/models")
        request = MagicMock(spec=Request)
        request.method = "GET"
        request.json = AsyncMock(side_effect=AssertionError("body must not be read for GET"))
        request.state = MagicMock()

        result = await validator("user@example.com", request)
        assert result is True
        mock_can_use.assert_called_once_with("my-endpoint", "user@example.com")
        request.json.assert_not_called()

    @pytest.mark.asyncio
    @patch("mlflow_oidc_auth.middleware.fastapi_permission_middleware.can_use_gateway_endpoint")
    async def test_gemini_route(self, mock_can_use):
        """Test Gemini generateContent route extracts endpoint name."""
        mock_can_use.return_value = True
        validator = self._get_gateway_validator("/gateway/gemini/v1beta/models/gemini-pro:generateContent")
        request = MagicMock(spec=Request)
        result = await validator("user@example.com", request)
        assert result is True
        mock_can_use.assert_called_once_with("gemini-pro", "user@example.com")


# ---------------------------------------------------------------------------
# Unit tests: OTel validator logic
# ---------------------------------------------------------------------------


class TestOtelValidator:
    """Test OTel trace ingestion validator."""

    def _get_otel_validator(self, path):
        from mlflow_oidc_auth.middleware.fastapi_permission_middleware import (
            _get_otel_validator,
        )

        return _get_otel_validator(path)

    @pytest.mark.asyncio
    @patch("mlflow_oidc_auth.utils.effective_experiment_permission")
    async def test_otel_with_experiment_id_allowed(self, mock_perm):
        """Test OTel with valid experiment ID and UPDATE permission."""
        mock_result = MagicMock()
        mock_result.permission.can_update = True
        mock_perm.return_value = mock_result

        validator = self._get_otel_validator("/v1/traces")
        request = MagicMock(spec=Request)
        request.headers = {"x-mlflow-experiment-id": "42"}

        result = await validator("user@example.com", request)
        assert result is True
        mock_perm.assert_called_once_with("42", "user@example.com")

    @pytest.mark.asyncio
    @patch("mlflow_oidc_auth.utils.effective_experiment_permission")
    async def test_otel_with_experiment_id_denied(self, mock_perm):
        """Test OTel with valid experiment ID but no UPDATE permission."""
        mock_result = MagicMock()
        mock_result.permission.can_update = False
        mock_perm.return_value = mock_result

        validator = self._get_otel_validator("/v1/traces")
        request = MagicMock(spec=Request)
        request.headers = {"x-mlflow-experiment-id": "42"}

        result = await validator("user@example.com", request)
        assert result is False

    @pytest.mark.asyncio
    async def test_otel_missing_experiment_header(self):
        """Test OTel without X-Mlflow-Experiment-Id header returns False."""
        validator = self._get_otel_validator("/v1/traces")
        request = MagicMock(spec=Request)
        request.headers = {}

        result = await validator("user@example.com", request)
        assert result is False


# ---------------------------------------------------------------------------
# Unit tests: require-authentication validator
# ---------------------------------------------------------------------------


class TestRequireAuthenticationValidator:
    """Test the simple authentication-only validator."""

    @pytest.mark.asyncio
    async def test_any_user_allowed(self):
        """Test that any authenticated user passes."""
        from mlflow_oidc_auth.middleware.fastapi_permission_middleware import (
            _get_require_authentication_validator,
        )

        validator = _get_require_authentication_validator()
        request = MagicMock(spec=Request)
        result = await validator("user@example.com", request)
        assert result is True


# ---------------------------------------------------------------------------
# Helper: create app with auth context injection + permission middleware
# ---------------------------------------------------------------------------


def _create_app_with_auth(username=None, is_admin=False, with_proxy_headers=False):
    """Create a test FastAPI app with auth context and permission middleware.

    Starlette ``@app.middleware("http")`` uses LIFO ordering: the last middleware
    registered wraps the outermost layer and runs first.  We must register the
    permission middleware FIRST, then the auth-context middleware, so that
    auth context is set before the permission middleware reads it.

    ``with_proxy_headers`` inserts the real ``ProxyHeadersMiddleware`` between the two,
    reproducing the production order Auth → ProxyHeaders → Permission.
    """
    from mlflow_oidc_auth.middleware.fastapi_permission_middleware import (
        add_fastapi_permission_middleware,
    )

    app = FastAPI()

    @app.get("/gateway/{endpoint_name}/mlflow/invocations")
    async def gateway_invocations(endpoint_name: str):
        return {"endpoint": endpoint_name}

    @app.get("/gateway/proxy/{endpoint_name}/user-agent")
    async def gateway_proxy_user_agent(endpoint_name: str, request: Request):
        return {
            "endpoint": endpoint_name,
            "user_agents": request.headers.getlist("user-agent"),
        }

    @app.get("/gateway/proxy/{endpoint_name}/{path:path}")
    async def gateway_proxy(endpoint_name: str, path: str):
        return {"endpoint": endpoint_name, "path": path}

    @app.post("/gateway/openai/v1/chat/completions")
    async def gateway_chat_completions(request: Request):
        return {"cached_body": getattr(request.state, "cached_body", None)}

    @app.get("/v1/traces")
    async def otel_traces():
        return {"traces": []}

    @app.get("/ajax-api/3.0/jobs")
    async def list_jobs():
        return {"jobs": []}

    @app.get("/ajax-api/3.0/mlflow/assistant/chat")
    async def assistant_chat():
        return {"response": "hello"}

    @app.get("/api/2.0/mlflow/experiments/list")
    async def flask_passthrough():
        return {"experiments": []}

    # Register permission middleware FIRST (will be inner)
    add_fastapi_permission_middleware(app)

    if with_proxy_headers:
        from mlflow_oidc_auth.middleware.proxy_headers_middleware import ProxyHeadersMiddleware

        app.add_middleware(ProxyHeadersMiddleware)

    # Register auth context LAST (will be outer — runs first)
    if username is not None:

        @app.middleware("http")
        async def inject_auth_context(request: Request, call_next):
            request.state.username = username
            request.state.is_admin = is_admin
            return await call_next(request)

    else:

        @app.middleware("http")
        async def emulate_unauthenticated_auth_middleware(request: Request, call_next):
            # Mirrors AuthMiddleware: admit credential-carrying gateway callers, recording
            # the decision so the permission middleware can re-check the routed path.
            from starlette.routing import get_route_path

            from mlflow_oidc_auth.utils.gateway_passthrough import (
                GATEWAY_PASSTHROUGH_SCOPE_KEY,
                allow_unauthenticated_gateway_call,
            )

            if allow_unauthenticated_gateway_call(get_route_path(request.scope), request.headers):
                request.scope[GATEWAY_PASSTHROUGH_SCOPE_KEY] = True

            return await call_next(request)

    return app


# ---------------------------------------------------------------------------
# Integration tests: full middleware with TestClient
# ---------------------------------------------------------------------------


class TestFastapiPermissionMiddlewareIntegration:
    """Integration tests for the middleware using FastAPI TestClient."""

    def _create_app_with_middleware(self):
        """Create a test app with no auth context (unauthenticated)."""
        return _create_app_with_auth(username=None)

    def test_unauthenticated_gateway_returns_401(self):
        """Test that gateway route without auth returns 401."""
        app = self._create_app_with_middleware()
        client = TestClient(app)
        response = client.get("/gateway/my-ep/mlflow/invocations")
        assert response.status_code == 401

    def test_unauthenticated_otel_returns_401(self):
        """Test that OTel route without auth returns 401."""
        app = self._create_app_with_middleware()
        client = TestClient(app)
        response = client.get("/v1/traces")
        assert response.status_code == 401

    def test_unauthenticated_jobs_returns_401(self):
        """Test that jobs route without auth returns 401."""
        app = self._create_app_with_middleware()
        client = TestClient(app)
        response = client.get("/ajax-api/3.0/jobs")
        assert response.status_code == 401

    def test_unauthenticated_assistant_returns_401(self):
        """Test that assistant route without auth returns 401."""
        app = self._create_app_with_middleware()
        client = TestClient(app)
        response = client.get("/ajax-api/3.0/mlflow/assistant/chat")
        assert response.status_code == 401

    def test_admin_gateway_passes(self):
        """Test that admin user passes gateway check."""
        app = _create_app_with_auth(username="admin@example.com", is_admin=True)
        client = TestClient(app)
        response = client.get("/gateway/my-ep/mlflow/invocations")
        assert response.status_code == 200
        assert response.json() == {"endpoint": "my-ep"}

    def test_admin_otel_passes(self):
        """Test that admin user passes OTel check."""
        app = _create_app_with_auth(username="admin@example.com", is_admin=True)
        client = TestClient(app)
        response = client.get("/v1/traces")
        assert response.status_code == 200

    def test_admin_jobs_passes(self):
        """Test that admin user passes jobs check."""
        app = _create_app_with_auth(username="admin@example.com", is_admin=True)
        client = TestClient(app)
        response = client.get("/ajax-api/3.0/jobs")
        assert response.status_code == 200

    @patch("mlflow_oidc_auth.middleware.fastapi_permission_middleware.can_use_gateway_endpoint")
    def test_regular_user_gateway_with_permission(self, mock_can_use):
        """Test that non-admin user with USE permission passes gateway."""
        mock_can_use.return_value = True
        app = _create_app_with_auth(username="user@example.com", is_admin=False)
        client = TestClient(app)
        response = client.get("/gateway/my-ep/mlflow/invocations")
        assert response.status_code == 200
        mock_can_use.assert_called_once_with("my-ep", "user@example.com")

    @patch("mlflow_oidc_auth.middleware.fastapi_permission_middleware.can_use_gateway_endpoint")
    def test_regular_user_gateway_without_permission(self, mock_can_use):
        """Test that non-admin user without USE permission gets 403."""
        mock_can_use.return_value = False
        app = _create_app_with_auth(username="user@example.com", is_admin=False)
        client = TestClient(app)
        response = client.get("/gateway/my-ep/mlflow/invocations")
        assert response.status_code == 403

    def test_flask_route_passes_through(self):
        """Test that Flask-handled routes pass through without permission check."""
        app = self._create_app_with_middleware()
        client = TestClient(app)
        response = client.get("/api/2.0/mlflow/experiments/list")
        assert response.status_code == 200

    def test_authenticated_user_jobs_passes(self):
        """Test that any authenticated user passes jobs check."""
        app = _create_app_with_auth(username="user@example.com", is_admin=False)
        client = TestClient(app)
        response = client.get("/ajax-api/3.0/jobs")
        assert response.status_code == 200

    def test_authenticated_user_assistant_passes(self):
        """Test that any authenticated user passes assistant check."""
        app = _create_app_with_auth(username="user@example.com", is_admin=False)
        client = TestClient(app)
        response = client.get("/ajax-api/3.0/mlflow/assistant/chat")
        assert response.status_code == 200

    @patch("mlflow_oidc_auth.utils.effective_experiment_permission")
    def test_regular_user_otel_with_update_permission(self, mock_perm):
        """Test that non-admin user with UPDATE permission passes OTel."""
        mock_result = MagicMock()
        mock_result.permission.can_update = True
        mock_perm.return_value = mock_result

        app = _create_app_with_auth(username="user@example.com", is_admin=False)
        client = TestClient(app)
        response = client.get("/v1/traces", headers={"X-Mlflow-Experiment-Id": "42"})
        assert response.status_code == 200
        mock_perm.assert_called_once_with("42", "user@example.com")

    @patch("mlflow_oidc_auth.utils.effective_experiment_permission")
    def test_regular_user_otel_without_update_permission(self, mock_perm):
        """Test that non-admin user without UPDATE permission gets 403."""
        mock_result = MagicMock()
        mock_result.permission.can_update = False
        mock_perm.return_value = mock_result

        app = _create_app_with_auth(username="user@example.com", is_admin=False)
        client = TestClient(app)
        response = client.get("/v1/traces", headers={"X-Mlflow-Experiment-Id": "42"})
        assert response.status_code == 403

    def test_regular_user_otel_missing_experiment_header(self):
        """Test OTel without experiment header gets 403."""
        app = _create_app_with_auth(username="user@example.com", is_admin=False)
        client = TestClient(app)
        response = client.get("/v1/traces")
        assert response.status_code == 403

    @patch("mlflow_oidc_auth.middleware.fastapi_permission_middleware.can_use_gateway_endpoint")
    def test_validator_exception_returns_403(self, mock_can_use):
        """Test that an exception in validator returns 403 (fail closed)."""
        mock_can_use.side_effect = RuntimeError("unexpected error")
        app = _create_app_with_auth(username="user@example.com", is_admin=False)
        client = TestClient(app)
        response = client.get("/gateway/my-ep/mlflow/invocations")
        assert response.status_code == 403

    @patch("mlflow_oidc_auth.middleware.fastapi_permission_middleware.allow_unauthenticated_gateway_call")
    def test_unauthenticated_gateway_passthrough_disabled_returns_401(self, mock_allow):
        """Test that an unauthenticated gateway call is rejected when pass-through is off."""
        mock_allow.return_value = False
        app = self._create_app_with_middleware()
        client = TestClient(app)
        response = client.get("/gateway/proxy/my-ep/models")
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Integration tests: unauthenticated end-user-authenticated endpoint allowlist
# ---------------------------------------------------------------------------


def _config(endpoints):
    """Build a config double exposing only the gateway allowlist."""
    config_mock = MagicMock()
    config_mock.OIDC_GATEWAY_END_USER_AUTH_ENDPOINTS = endpoints
    return config_mock


class TestUnauthenticatedGatewayAllowlist:
    """Test the per-endpoint allowlist for callers with no OIDC identity."""

    _CLI_HEADERS = {"user-agent": "claude-cli/1.0.99 (external, cli)", "Authorization": "Bearer upstream-token"}

    def _client(self):
        return TestClient(_create_app_with_auth(username=None))

    def test_url_derived_listed_endpoint_passes(self):
        """Test a listed URL-derived endpoint reaches the handler."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
            response = self._client().get("/gateway/proxy/my-ep/models", headers=self._CLI_HEADERS)

        assert response.status_code == 200
        assert response.json() == {"endpoint": "my-ep", "path": "models"}

    def test_url_derived_unlisted_endpoint_returns_401(self):
        """Test an unlisted URL-derived endpoint is rejected despite credentials."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
            response = self._client().get("/gateway/proxy/other-ep/models", headers=self._CLI_HEADERS)

        assert response.status_code == 401

    def test_url_derived_listed_endpoint_without_credentials_returns_401(self):
        """Test a listed endpoint still needs the caller's own credential header."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
            response = self._client().get("/gateway/proxy/my-ep/models")

        assert response.status_code == 401

    def test_url_derived_empty_allowlist_returns_401(self):
        """Test an empty allowlist rejects even a would-be listed endpoint."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config([])):
            response = self._client().get("/gateway/proxy/my-ep/models", headers=self._CLI_HEADERS)

        assert response.status_code == 401

    def test_body_derived_listed_endpoint_passes_and_caches_body(self):
        """Test a listed body-derived endpoint passes and leaves the body readable."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-model"])):
            response = self._client().post(
                "/gateway/openai/v1/chat/completions",
                json={"model": "my-model"},
                headers=self._CLI_HEADERS,
            )

        assert response.status_code == 200
        assert response.json() == {"cached_body": {"model": "my-model"}}

    def test_body_derived_unlisted_endpoint_returns_401(self):
        """Test an unlisted body-derived endpoint is rejected despite credentials."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-model"])):
            response = self._client().post(
                "/gateway/openai/v1/chat/completions",
                json={"model": "other-model"},
                headers=self._CLI_HEADERS,
            )

        assert response.status_code == 401

    def test_body_derived_listed_endpoint_without_credentials_returns_401(self):
        """Test a listed body-derived endpoint still needs a credential header."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-model"])):
            response = self._client().post("/gateway/openai/v1/chat/completions", json={"model": "my-model"})

        assert response.status_code == 401

    def test_body_derived_empty_allowlist_returns_401(self):
        """Test an empty allowlist rejects a body-derived endpoint too."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config([])):
            response = self._client().post(
                "/gateway/openai/v1/chat/completions",
                json={"model": "my-model"},
                headers=self._CLI_HEADERS,
            )

        assert response.status_code == 401

    @patch("mlflow_oidc_auth.middleware.fastapi_permission_middleware.can_use_gateway_endpoint")
    def test_authenticated_user_still_needs_use_permission_on_listed_endpoint(self, mock_can_use):
        """Test designation does not disable authorization for callers with an identity."""
        mock_can_use.return_value = False
        app = _create_app_with_auth(username="user@example.com", is_admin=False)
        client = TestClient(app)

        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
            response = client.get("/gateway/proxy/my-ep/models", headers=self._CLI_HEADERS)

        assert response.status_code == 403
        mock_can_use.assert_called_once_with("my-ep", "user@example.com")


# ---------------------------------------------------------------------------
# Integration tests: the caller's User-Agent must reach MLflow unmodified
# ---------------------------------------------------------------------------


class TestClientUserAgentIsForwardedUnmodified:
    """Test that admitted callers reach MLflow with their request unmodified."""

    _CLI_HEADERS = {"user-agent": "claude-cli/1.0.99 (external, cli)", "Authorization": "Bearer upstream-token"}
    _ODD_UA_HEADERS = {"user-agent": "weird-agent/9", "Authorization": "Bearer upstream-token"}

    def _client(self, username=None, is_admin=False):
        return TestClient(_create_app_with_auth(username=username, is_admin=is_admin))

    def test_listed_endpoint_reaches_handler_with_user_agent_unmodified(self):
        """Test an allowed unauthenticated call arrives upstream exactly as sent."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
            response = self._client().get("/gateway/proxy/my-ep/user-agent", headers=self._CLI_HEADERS)

        assert response.status_code == 200
        assert response.json()["user_agents"] == [self._CLI_HEADERS["user-agent"]]

    def test_unlisted_endpoint_returns_401_and_never_reaches_handler(self):
        """Test an unlisted endpoint is rejected before the handler runs."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
            response = self._client().get("/gateway/proxy/other-ep/user-agent", headers=self._CLI_HEADERS)

        assert response.status_code == 401
        assert "user_agents" not in response.json()

    def test_unrecognised_user_agent_returns_401(self):
        """Test a caller MLflow would serve the server key to is rejected outright."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
            response = self._client().get("/gateway/proxy/my-ep/user-agent", headers=self._ODD_UA_HEADERS)

        assert response.status_code == 401
        assert "user_agents" not in response.json()

    def test_missing_user_agent_returns_401(self):
        """Test a credential with no User-Agent is not enough to bypass authentication."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
            response = self._client().get("/gateway/proxy/my-ep/user-agent", headers={"Authorization": "Bearer upstream-token"})

        assert response.status_code == 401

    @patch("mlflow_oidc_auth.middleware.fastapi_permission_middleware.can_use_gateway_endpoint")
    def test_authenticated_user_user_agent_is_untouched(self, mock_can_use):
        """Test an authenticated caller keeps using the server-side key."""
        mock_can_use.return_value = True
        response = self._client(username="user@example.com").get("/gateway/proxy/my-ep/user-agent", headers=self._ODD_UA_HEADERS)

        assert response.status_code == 200
        assert response.json()["user_agents"] == ["weird-agent/9"]


# ---------------------------------------------------------------------------
# Integration tests: X-Forwarded-Prefix must not shift the authorized path
# ---------------------------------------------------------------------------


class TestForwardedPrefixCannotBypassAuthorization:
    """Test that a client-supplied X-Forwarded-Prefix cannot move the routed path
    out of reach of the permission middleware."""

    _CLI_HEADERS = {"user-agent": "claude-cli/1.0.99 (external, cli)", "Authorization": "Bearer upstream-token"}
    _PREFIX = {"X-Forwarded-Prefix": "/proxied"}

    def test_unauthenticated_unlisted_endpoint_behind_prefix_returns_401(self):
        """Test a prefix-shifted path to an unlisted endpoint is still rejected."""
        client = TestClient(_create_app_with_auth(username=None, with_proxy_headers=True))

        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
            response = client.get("/proxied/gateway/proxy/other-ep/models", headers={**self._CLI_HEADERS, **self._PREFIX})

        assert response.status_code == 401

    @patch("mlflow_oidc_auth.middleware.fastapi_permission_middleware.can_use_gateway_endpoint")
    def test_authenticated_unauthorized_endpoint_behind_prefix_returns_403(self, mock_can_use):
        """Test a prefix-shifted path still runs the per-endpoint permission check."""
        mock_can_use.return_value = False
        client = TestClient(_create_app_with_auth(username="user@example.com", is_admin=False, with_proxy_headers=True))

        response = client.get("/proxied/gateway/proxy/my-ep/models", headers=self._PREFIX)

        assert response.status_code == 403
        mock_can_use.assert_called_once_with("my-ep", "user@example.com")


# ---------------------------------------------------------------------------
# Integration tests: real AuthMiddleware + ProxyHeaders + permission middleware
# ---------------------------------------------------------------------------


def _create_app_with_real_auth_middleware():
    """Wire the middleware exactly as app.create_app does: Auth → ProxyHeaders → Permission."""
    from mlflow_oidc_auth.middleware.auth_middleware import AuthMiddleware
    from mlflow_oidc_auth.middleware.fastapi_permission_middleware import (
        add_fastapi_permission_middleware,
    )
    from mlflow_oidc_auth.middleware.proxy_headers_middleware import ProxyHeadersMiddleware

    app = FastAPI()

    @app.get("/gateway/proxy/{endpoint_name}/{path:path}")
    async def gateway_proxy(endpoint_name: str, path: str):
        return {"endpoint": endpoint_name, "path": path}

    @app.get("/api/2.0/mlflow/permissions/users")
    async def flask_style_route(request: Request):
        return {"reached": True, "username": getattr(request.state, "username", None)}

    add_fastapi_permission_middleware(app)
    app.add_middleware(ProxyHeadersMiddleware)
    app.add_middleware(AuthMiddleware)

    return app


class TestForwardedPrefixAgainstRealAuthMiddleware:
    """AuthMiddleware runs outside ProxyHeadersMiddleware and so judges the path before the
    client-supplied X-Forwarded-Prefix is stripped. Its admission must not carry over to
    whichever route that prefix finally selects."""

    _CLI_HEADERS = {"user-agent": "claude-cli/1.0.99 (external, cli)", "Authorization": "Bearer upstream-token"}

    def _client(self):
        return TestClient(_create_app_with_real_auth_middleware(), raise_server_exceptions=False)

    def test_listed_endpoint_is_admitted_without_oidc_identity(self):
        """Test the feature still works end to end through the real auth stack."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
            response = self._client().get("/gateway/proxy/my-ep/models", headers=self._CLI_HEADERS)

        assert response.status_code == 200
        assert response.json() == {"endpoint": "my-ep", "path": "models"}

    def test_prefix_cannot_divert_an_admitted_call_to_another_route(self):
        """Test a prefix that reroutes an admitted gateway call elsewhere is rejected."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
            response = self._client().get(
                "/gateway/proxy/my-ep/api/2.0/mlflow/permissions/users",
                headers={**self._CLI_HEADERS, "X-Forwarded-Prefix": "/gateway/proxy/my-ep"},
            )

        assert response.status_code == 401
        assert "reached" not in response.json()


# ---------------------------------------------------------------------------
# Integration tests: malformed bodies must not reach an unhandled exception
# ---------------------------------------------------------------------------


class TestMalformedBodyOnUnauthenticatedGatewayCall:
    """Test that an unauthenticated caller cannot turn a bad body into a 500."""

    _CLI_HEADERS = {"user-agent": "claude-cli/1.0.99 (external, cli)", "Authorization": "Bearer upstream-token"}

    @pytest.mark.parametrize(
        "body",
        [
            {"model": {"a": 1}},
            {"model": ["x"]},
            {"model": 123},
            ["not", "a", "dict"],
            "juststring",
        ],
    )
    def test_malformed_body_returns_401(self, body):
        """Test each malformed body is rejected with 401 rather than crashing."""
        client = TestClient(_create_app_with_auth(username=None))

        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-model"])):
            response = client.post("/gateway/openai/v1/chat/completions", json=body, headers=self._CLI_HEADERS)

        assert response.status_code == 401

    def test_unexpected_allowlist_failure_returns_401(self):
        """Test an unexpected error in the allowlist check fails closed."""
        client = TestClient(_create_app_with_auth(username=None))

        with patch(
            "mlflow_oidc_auth.middleware.fastapi_permission_middleware.is_end_user_auth_endpoint",
            side_effect=RuntimeError("boom"),
        ):
            with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
                response = client.get("/gateway/proxy/my-ep/models", headers=self._CLI_HEADERS)

        assert response.status_code == 401
