"""
FastAPI Permission Middleware for MLflow OIDC Auth.

This middleware enforces authorization on FastAPI-native routes (gateway invocations,
OTel trace ingestion, assistant, job API) that bypass Flask and therefore bypass
the Flask ``before_request_hook``.

It mirrors the upstream ``add_fastapi_permission_middleware`` from
``mlflow/server/auth/__init__.py`` but uses our OIDC-based authentication context
(set by ``AuthMiddleware``) instead of upstream's Basic-Auth-only approach.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import get_route_path

from mlflow_oidc_auth.logger import get_logger
from mlflow_oidc_auth.utils.gateway_passthrough import (
    GATEWAY_PASSTHROUGH_SCOPE_KEY,
    ROUTES_NEEDING_BODY,
    allow_unauthenticated_gateway_call,
    endpoint_name_from_path,
    is_end_user_auth_endpoint,
)
from mlflow_oidc_auth.utils.permissions import can_use_gateway_endpoint

logger = get_logger()


# ---------------------------------------------------------------------------
# Endpoint-name extraction (mirrors upstream _extract_gateway_endpoint_name)
# ---------------------------------------------------------------------------


def _extract_gateway_endpoint_name(path: str, body: Any) -> str | None:
    """Extract the target endpoint name from the URL, or from the body for routes carrying it there."""
    if name := endpoint_name_from_path(path):
        return name

    if path in ROUTES_NEEDING_BODY and isinstance(body, dict):
        model = body.get("model")

        return model if isinstance(model, str) else None

    return None


async def _resolve_gateway_endpoint_name(path: str, request: Request) -> str | None:
    """Resolve the target endpoint name, reading and caching the body when required."""
    body: Any = None
    if path in ROUTES_NEEDING_BODY and request.method not in ("GET", "HEAD"):
        try:
            body = await request.json()
            # Starlette allows the body to be read only once.
            request.state.cached_body = body
        except Exception:
            return None

    return _extract_gateway_endpoint_name(path, body)


async def _is_designated_end_user_auth_call(path: str, request: Request) -> bool:
    """Check whether an unauthenticated gateway call targets a designated end-user-auth endpoint."""
    try:
        if not allow_unauthenticated_gateway_call(path, request.headers):
            return False

        endpoint_name = await _resolve_gateway_endpoint_name(path, request)

        return endpoint_name is not None and is_end_user_auth_endpoint(endpoint_name)
    except Exception:
        logger.exception("Unauthenticated gateway check failed for path %s", path)

        return False


def _authentication_required_response() -> JSONResponse:
    """Build the 401 returned to callers with no usable identity."""
    return JSONResponse(
        status_code=401,
        content={"detail": "Authentication required"},
        headers={"WWW-Authenticate": 'Basic realm="mlflow"'},
    )


# ---------------------------------------------------------------------------
# Per-route validator factories
# ---------------------------------------------------------------------------


def _get_gateway_validator(
    path: str,
) -> Callable[[str, Request], Awaitable[bool]] | None:
    """Return an async validator for gateway invocation routes.

    Validates that the user has USE permission on the target gateway endpoint.
    """

    async def validator(username: str, request: Request) -> bool:
        endpoint_name = await _resolve_gateway_endpoint_name(path, request)
        if endpoint_name is None:
            logger.warning("Gateway validator: no endpoint name found in request path %s", path)
            return False

        return can_use_gateway_endpoint(endpoint_name, username)

    return validator


def _get_otel_validator(path: str) -> Callable[[str, Request], Awaitable[bool]] | None:
    """Return an async validator for OTel trace ingestion routes.

    Requires UPDATE permission on the experiment identified by the
    ``X-Mlflow-Experiment-Id`` header.
    """

    async def validator(username: str, request: Request) -> bool:
        from mlflow_oidc_auth.utils import effective_experiment_permission

        experiment_id = request.headers.get("x-mlflow-experiment-id")
        if not experiment_id:
            logger.warning("OTel validator: missing X-Mlflow-Experiment-Id header")
            return False

        return effective_experiment_permission(experiment_id, username).permission.can_update

    return validator


def _get_require_authentication_validator() -> Callable[[str, Request], Awaitable[bool]]:
    """Return a validator that allows any authenticated user."""

    async def validator(username: str, request: Request) -> bool:
        return True

    return validator


# ---------------------------------------------------------------------------
# Route → validator dispatcher
# ---------------------------------------------------------------------------


def _find_fastapi_validator(
    path: str,
) -> Callable[[str, Request], Awaitable[bool]] | None:
    """Find the validator for a FastAPI-native route.

    Returns a validator function for routes that need permission checks, or
    ``None`` if the route should be handled by Flask (WSGI fall-through).
    """
    if path.startswith("/gateway/"):
        return _get_gateway_validator(path)

    if path.startswith("/v1/traces"):
        return _get_otel_validator(path)

    if path.startswith("/ajax-api/3.0/jobs"):
        return _get_require_authentication_validator()

    if path.startswith("/ajax-api/3.0/mlflow/assistant"):
        return _get_require_authentication_validator()

    return None


# ---------------------------------------------------------------------------
# Middleware registration
# ---------------------------------------------------------------------------


def add_fastapi_permission_middleware(app: FastAPI) -> None:
    """Add OIDC-aware permission middleware for FastAPI-native routes.

    This middleware runs AFTER ``AuthMiddleware`` (which has already set
    ``request.state.username`` / ``request.state.is_admin`` and the ASGI
    scope ``mlflow_oidc_auth`` dict).  It only activates for routes served
    directly by FastAPI (gateway, otel, assistant, job API) — all other
    requests fall through to the Flask WSGI mount where the Flask hooks
    handle authorization.
    """

    @app.middleware("http")
    async def fastapi_permission_middleware(request: Request, call_next):
        # Authorize on the path Starlette actually routes on, not the external path:
        # ProxyHeadersMiddleware derives root_path from the client-supplied X-Forwarded-Prefix.
        path = get_route_path(request.scope)

        # Check authentication context (already set by AuthMiddleware)
        username = getattr(request.state, "username", None)

        # AuthMiddleware runs outside ProxyHeadersMiddleware, so it judged the path before the
        # client-supplied X-Forwarded-Prefix was stripped. Re-check what it admitted against
        # the path actually routed, or that prefix could divert the call to any other route.
        if not username and request.scope.get(GATEWAY_PASSTHROUGH_SCOPE_KEY):
            if await _is_designated_end_user_auth_call(path, request):
                return await call_next(request)

            logger.info("Unauthenticated gateway call rejected for path %s", path)

            return _authentication_required_response()

        # Find validator for this route — returns None for Flask-handled routes
        validator = _find_fastapi_validator(path)
        if validator is None:
            return await call_next(request)

        if not username:
            logger.info("Unauthenticated request rejected for path %s", path)

            return _authentication_required_response()

        # Admins have full access
        is_admin = getattr(request.state, "is_admin", False)
        if is_admin:
            return await call_next(request)

        # Run the validator
        try:
            if not await validator(username, request):
                return PlainTextResponse(
                    "Permission denied",
                    status_code=403,
                )
        except Exception as e:
            logger.error("FastAPI permission middleware error: %s", type(e).__name__)
            return PlainTextResponse(
                "Permission denied",
                status_code=403,
            )

        return await call_next(request)
