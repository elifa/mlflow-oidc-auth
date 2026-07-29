"""
Gateway pass-through helpers for callers that supply their own provider credentials.

Coding agents such as the GitHub Copilot CLI authenticate to the MLflow AI Gateway with
their own upstream provider credentials and cannot perform an OIDC login. The upstream
provider authenticates them, not this plugin, so there is no OIDC identity to authorise
against and the per-endpoint USE permission check cannot be applied.

Letting such calls through is therefore opt-in via the
``OIDC_GATEWAY_END_USER_AUTH_ENDPOINTS`` allowlist and limited to the gateway invocation
routes; only the designated endpoints accept callers with no OIDC identity, and gateway
management APIs always require a full OIDC identity.

Admission is further limited to callers MLflow itself recognises as supplying their own
credentials. MLflow substitutes the endpoint's configured server key for everyone else, so
admitting a caller it does not recognise would turn the endpoint into an open proxy onto
that key.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Final

from mlflow_oidc_auth.config import config
from mlflow_oidc_auth.logger import get_logger

logger = get_logger()

# ASGI scope key marking a request that AuthMiddleware admitted with no OIDC identity.
GATEWAY_PASSTHROUGH_SCOPE_KEY: Final = "mlflow_oidc_auth.gateway_passthrough"

# Gateway invocation ("use") routes that name their target endpoint in the request body
# rather than the URL. Management APIs — /api/2.0/gateway/..., /api/3.0/gateway/... and
# the /ajax-api/... console routes — are deliberately absent: they always require a full
# OIDC identity.
ROUTES_NEEDING_BODY: Final = frozenset(
    (
        "/gateway/mlflow/v1/chat/completions",
        "/gateway/openai/v1/chat/completions",
        "/gateway/openai/v1/embeddings",
        "/gateway/openai/v1/responses",
        "/gateway/openai/v1/responses/compact",
        "/gateway/anthropic/v1/messages",
    )
)

# Gateway invocation routes that name their target endpoint in the URL. Each pattern must
# capture that name as group 1.
_ENDPOINT_NAME_PATTERNS: Final = (
    re.compile(r"^/gateway/([^/]+)/mlflow/invocations$"),
    re.compile(r"^/gateway/proxy/([^/]+)(?:/.*)?$"),
    re.compile(r"^/gateway/gemini/v1beta/models/([^/:]+):generateContent$"),
    re.compile(r"^/gateway/gemini/v1beta/models/([^/:]+):streamGenerateContent$"),
)


def is_gateway_use_path(path: str) -> bool:
    """Check whether a request path targets an MLflow AI Gateway invocation route.

    Args:
        path: Request path, e.g. ``/gateway/proxy/my-endpoint/models``.

    Returns:
        True if the path invokes a gateway endpoint, False otherwise. Gateway
        management APIs always return False.
    """
    if path in ROUTES_NEEDING_BODY:
        return True

    return any(pattern.match(path) for pattern in _ENDPOINT_NAME_PATTERNS)


def endpoint_name_from_path(path: str) -> str | None:
    """Extract the target gateway endpoint name from the request path.

    Args:
        path: Request path.

    Returns:
        The endpoint name, or None when the path does not carry one — either because it is
        not a gateway invocation route or because the name is in the request body.
    """
    for pattern in _ENDPOINT_NAME_PATTERNS:
        if match := pattern.match(path):
            return match.group(1)

    return None


@lru_cache(maxsize=1)
def _mlflow_client_auth_api() -> tuple[Callable[[Mapping[str, str]], bool], tuple[str, ...]] | None:
    """Resolve MLflow's client-credential API, importing it at most once.

    Returns:
        MLflow's ``_client_provides_auth`` predicate paired with its credential header
        names, or None when MLflow no longer exposes them.
    """
    # Imported lazily: mlflow.gateway.providers pulls in every provider (~1.3s), and by the
    # time a gateway request arrives the server has already paid that cost.
    try:
        from mlflow.gateway.providers.base import _CLIENT_AUTH_HEADERS, _client_provides_auth
    except (ImportError, AttributeError):
        # Private API, so treat its removal as a supported outcome: fail closed rather than
        # 500, and let these callers fall back to requiring an OIDC identity.
        logger.warning(
            "MLflow no longer exposes mlflow.gateway.providers.base._client_provides_auth; "
            "gateway callers supplying their own provider credentials now require an OIDC identity."
        )

        return None

    return _client_provides_auth, tuple(_CLIENT_AUTH_HEADERS)


def request_carries_end_user_credentials(headers: Mapping[str, str]) -> bool:
    """Check whether the caller supplies its own upstream provider credentials.

    Both the recognised User-Agents and the credential header names come from MLflow rather
    than a local copy: MLflow forwards the caller's credential only for requests it accepts
    here and otherwise substitutes the endpoint's server key, so a copy that drifted wider
    would admit callers whose requests are then billed to that key. Deferring keeps the two
    in lockstep, including across forks that extend the allowlist.

    Args:
        headers: Request headers (Starlette ``Headers`` or any mapping).

    Returns:
        True if MLflow recognises the caller as supplying its own credentials and a
        credential header is present with a non-empty value.
    """
    mlflow_api = _mlflow_client_auth_api()
    if mlflow_api is None:
        return False

    client_provides_auth, credential_headers = mlflow_api

    # A plain dict, unlike Starlette's Headers, is case-sensitive.
    normalized = {key.lower(): value for key, value in headers.items()}

    if not client_provides_auth(normalized):
        return False

    # Stricter than MLflow, which accepts a present-but-empty credential header.
    return any((normalized.get(header) or "").strip() for header in credential_headers)


def _configured_end_user_auth_endpoints() -> set[str]:
    """Return the configured allowlist, stripped of whitespace and empty entries.

    Returns:
        Set of endpoint names. Empty when the setting is unset or explicitly blank —
        ``OIDC_GATEWAY_END_USER_AUTH_ENDPOINTS=""`` parses to ``[""]``, which must not
        count as a designated endpoint.
    """
    configured = {entry.strip() for entry in config.OIDC_GATEWAY_END_USER_AUTH_ENDPOINTS}
    configured.discard("")

    return configured


def is_end_user_auth_endpoint(endpoint_name: str) -> bool:
    """Check whether a gateway endpoint is designated as end-user-authenticated.

    Args:
        endpoint_name: Target gateway endpoint name.

    Returns:
        True if the name exactly matches a configured entry. Matching is case-sensitive
        because MLflow endpoint names are.
    """
    return endpoint_name in _configured_end_user_auth_endpoints()


def allow_unauthenticated_gateway_call(path: str, headers: Mapping[str, str]) -> bool:
    """Check whether a gateway request may proceed without an OIDC identity.

    Routes naming their endpoint in the body are admitted on the route alone, because
    ``AuthMiddleware`` runs before the body can safely be read; ``fastapi_permission_middleware``
    checks those against the allowlist once the body is available.

    Args:
        path: Request path.
        headers: Request headers.

    Returns:
        True only when the allowlist is non-empty, the path targets a gateway invocation
        route for a designated endpoint, and the caller carries its own upstream credentials.
    """
    if not _configured_end_user_auth_endpoints():
        return False

    if not is_gateway_use_path(path):
        return False

    endpoint_name = endpoint_name_from_path(path)
    if endpoint_name is not None and not is_end_user_auth_endpoint(endpoint_name):
        return False

    return request_carries_end_user_credentials(headers)
