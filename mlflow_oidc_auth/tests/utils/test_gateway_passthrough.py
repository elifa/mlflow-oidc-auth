"""
Tests for gateway pass-through helpers.

Covers gateway invocation path detection, credential header detection, and the
``OIDC_GATEWAY_END_USER_AUTH_ENDPOINTS`` allowlist.
"""

from unittest.mock import MagicMock, patch

import pytest

from mlflow_oidc_auth.utils.gateway_passthrough import (
    ROUTES_NEEDING_BODY,
    _mlflow_client_auth_api,
    allow_unauthenticated_gateway_call,
    is_end_user_auth_endpoint,
    is_gateway_use_path,
    request_carries_end_user_credentials,
)


def test_unrecognised_user_agent_is_rejected():
    """A User-Agent MLflow does not recognise would be served the server key, so reject it."""
    headers = {"user-agent": "weird-agent/9", "Authorization": "Bearer upstream-token"}

    assert request_carries_end_user_credentials(headers) is False


def test_missing_mlflow_private_api_fails_closed():
    """Test the feature switches itself off instead of raising if MLflow drops the API."""
    headers = {"user-agent": "claude-cli/1.0.99 (external, cli)", "Authorization": "Bearer upstream-token"}

    _mlflow_client_auth_api.cache_clear()
    try:
        with patch("mlflow_oidc_auth.utils.gateway_passthrough._mlflow_client_auth_api", return_value=None):
            assert request_carries_end_user_credentials(headers) is False
    finally:
        _mlflow_client_auth_api.cache_clear()


class TestIsGatewayUsePath:
    """Test gateway invocation path detection."""

    @pytest.mark.parametrize("path", sorted(ROUTES_NEEDING_BODY))
    def test_literal_use_paths(self, path):
        """Test every literal invocation route is recognised."""
        assert is_gateway_use_path(path) is True

    def test_invocations_path(self):
        """Test an invocations route is recognised."""
        assert is_gateway_use_path("/gateway/my-endpoint/mlflow/invocations") is True

    def test_proxy_path(self):
        """Test a proxy route is recognised."""
        assert is_gateway_use_path("/gateway/proxy/my-endpoint/models") is True

    def test_proxy_path_without_trailing_path(self):
        """Test a proxy route without a trailing path is recognised."""
        assert is_gateway_use_path("/gateway/proxy/my-endpoint") is True

    def test_proxy_path_with_nested_path(self):
        """Test a proxy route with a nested path is recognised."""
        assert is_gateway_use_path("/gateway/proxy/my-endpoint/v1/chat/completions") is True

    def test_gemini_generate_content(self):
        """Test the Gemini generateContent route is recognised."""
        assert is_gateway_use_path("/gateway/gemini/v1beta/models/my-endpoint:generateContent") is True

    def test_gemini_stream_generate_content(self):
        """Test the Gemini streamGenerateContent route is recognised."""
        assert is_gateway_use_path("/gateway/gemini/v1beta/models/my-endpoint:streamGenerateContent") is True

    @pytest.mark.parametrize(
        "path",
        [
            "/api/2.0/gateway/routes/my-endpoint",
            "/api/3.0/gateway/endpoint/my-endpoint",
            "/api/3.0/gateway/route/my-endpoint",
            "/api/2.0/gateway/limits/my-endpoint",
            "/ajax-api/2.0/mlflow/gateway/supported-providers",
            "/ajax-api/2.0/mlflow/gateway/provider-config",
            "/ajax-api/2.0/mlflow/gateway/secrets-config",
            "/api/2.0/mlflow/permissions/gateways/endpoints/my-endpoint",
        ],
    )
    def test_management_paths_are_not_use_paths(self, path):
        """Test gateway management APIs are never treated as invocation routes."""
        assert is_gateway_use_path(path) is False

    @pytest.mark.parametrize(
        "path",
        [
            "/gateway/",
            "/gateway",
            "/gatewayfoo/bar",
            "/gateway/my-endpoint/mlflow/invocations/extra",
            "/gateway/openai/v1/responses/compact/extra",
            "/api/2.0/mlflow/experiments/list",
        ],
    )
    def test_non_use_paths(self, path):
        """Test unrelated or over-long paths are not invocation routes."""
        assert is_gateway_use_path(path) is False


class TestRequestCarriesEndUserCredentials:
    """Test credential detection: a recognised CLI User-Agent plus a credential header."""

    # Recognised by both upstream MLflow and forks that extend the allowlist.
    _AGENT = "claude-cli/1.0.99 (external, cli)"

    def test_authorization_header(self):
        """Test an Authorization header is recognised."""
        assert request_carries_end_user_credentials({"user-agent": self._AGENT, "authorization": "Bearer tok"}) is True

    def test_authorization_header_mixed_case_key(self):
        """Test header lookup is case-insensitive for plain dicts."""
        assert request_carries_end_user_credentials({"User-Agent": self._AGENT, "Authorization": "Bearer tok"}) is True

    @pytest.mark.parametrize("header", ["x-api-key", "x-goog-api-key", "api-key"])
    def test_provider_api_key_headers(self, header):
        """Test each provider API key header is recognised."""
        assert request_carries_end_user_credentials({"user-agent": self._AGENT, header: "k"}) is True

    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_credential_value(self, value):
        """Test an empty or whitespace-only credential counts as absent."""
        assert request_carries_end_user_credentials({"user-agent": self._AGENT, "authorization": value}) is False

    def test_agent_user_agent_without_auth_header(self):
        """Test a coding-agent User-Agent alone is not enough."""
        assert request_carries_end_user_credentials({"user-agent": self._AGENT}) is False

    def test_empty_headers(self):
        """Test empty headers are not recognised."""
        assert request_carries_end_user_credentials({}) is False

    def test_browser_user_agent_with_authorization(self):
        """Test a browser session token is not mistaken for an upstream credential."""
        headers = {"user-agent": "Mozilla/5.0 (X11; Linux x86_64)", "Authorization": "Bearer session-token"}
        assert request_carries_end_user_credentials(headers) is False


def _config(endpoints: list[str]) -> MagicMock:
    config_mock = MagicMock()
    config_mock.OIDC_GATEWAY_END_USER_AUTH_ENDPOINTS = endpoints
    return config_mock


class TestIsEndUserAuthEndpoint:
    """Test the per-endpoint allowlist lookup."""

    def test_exact_match(self):
        """Test a listed endpoint name matches."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep", "other-ep"])):
            assert is_end_user_auth_endpoint("my-ep") is True

    def test_not_listed(self):
        """Test an unlisted endpoint name does not match."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
            assert is_end_user_auth_endpoint("other-ep") is False

    def test_empty_allowlist(self):
        """Test an empty allowlist matches nothing."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config([])):
            assert is_end_user_auth_endpoint("my-ep") is False

    def test_surrounding_whitespace_is_stripped(self):
        """Test entries from a comma-separated env var still match."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["  my-ep  ", "\tother-ep\n"])):
            assert is_end_user_auth_endpoint("my-ep") is True
            assert is_end_user_auth_endpoint("other-ep") is True

    def test_empty_entries_are_ignored(self):
        """Test blank entries never match a blank endpoint name."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["", "   ", "my-ep"])):
            assert is_end_user_auth_endpoint("") is False
            assert is_end_user_auth_endpoint("   ") is False
            assert is_end_user_auth_endpoint("my-ep") is True

    def test_matching_is_case_sensitive(self):
        """Test matching is case-sensitive, as MLflow endpoint names are."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["My-EP"])):
            assert is_end_user_auth_endpoint("My-EP") is True
            assert is_end_user_auth_endpoint("my-ep") is False


class TestAllowUnauthenticatedGatewayCall:
    """Test the combination of allowlist, path and credential checks."""

    _CLI_HEADERS = {"user-agent": "claude-cli/1.0.99 (external, cli)", "Authorization": "Bearer upstream-token"}

    def test_allowed_when_allowlist_non_empty(self):
        """Test a credential-carrying gateway call to a listed endpoint is allowed."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
            assert allow_unauthenticated_gateway_call("/gateway/proxy/my-ep/models", self._CLI_HEADERS) is True

    def test_denied_when_allowlist_empty(self):
        """Test an empty allowlist gates the whole feature off."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config([])):
            assert allow_unauthenticated_gateway_call("/gateway/proxy/my-ep/models", self._CLI_HEADERS) is False

    def test_denied_when_allowlist_is_explicitly_blank(self):
        """Test OIDC_GATEWAY_END_USER_AUTH_ENDPOINTS="" parses to [""] and stays off."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config([""])):
            assert allow_unauthenticated_gateway_call("/gateway/proxy/my-ep/models", self._CLI_HEADERS) is False

    def test_denied_when_allowlist_is_only_whitespace(self):
        """Test whitespace-only entries do not enable the feature either."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["  ", ""])):
            assert allow_unauthenticated_gateway_call("/gateway/proxy/my-ep/models", self._CLI_HEADERS) is False

    @pytest.mark.parametrize(
        "path",
        [
            "/gateway/proxy/unlisted-ep/models",
            "/gateway/unlisted-ep/mlflow/invocations",
            "/gateway/gemini/v1beta/models/unlisted-ep:generateContent",
            "/gateway/gemini/v1beta/models/unlisted-ep:streamGenerateContent",
        ],
    )
    def test_denied_for_unlisted_endpoint_named_in_the_url(self, path):
        """Test an endpoint name available from the URL is checked before admission."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["listed-ep"])):
            assert allow_unauthenticated_gateway_call(path, self._CLI_HEADERS) is False

    @pytest.mark.parametrize("path", sorted(ROUTES_NEEDING_BODY))
    def test_body_derived_routes_are_admitted_pending_the_middleware_check(self, path):
        """Test routes naming their endpoint in the body pass here; the middleware checks them."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["listed-ep"])):
            assert allow_unauthenticated_gateway_call(path, self._CLI_HEADERS) is True

    def test_denied_for_non_gateway_path(self):
        """Test non-gateway paths are never allowed through."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
            assert allow_unauthenticated_gateway_call("/api/2.0/mlflow/experiments/list", self._CLI_HEADERS) is False

    def test_denied_for_gateway_management_path(self):
        """Test gateway management APIs are rejected even with valid credentials."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
            assert allow_unauthenticated_gateway_call("/api/2.0/gateway/routes/my-endpoint", self._CLI_HEADERS) is False

    def test_denied_without_client_credentials(self):
        """Test gateway calls without client credentials are not allowed through."""
        with patch("mlflow_oidc_auth.utils.gateway_passthrough.config", _config(["my-ep"])):
            headers = {"user-agent": "Mozilla/5.0 (X11; Linux x86_64)"}
            assert allow_unauthenticated_gateway_call("/gateway/proxy/my-ep/models", headers) is False
