"""OAuth configuration for the FastAPI application.

Unit tests expect the module attribute `oauth` to be an instance of
`authlib.integrations.starlette_client.OAuth`.

We keep OIDC client registration lazy so importing this module does not require
OIDC configuration to be present (and does not perform network calls).
"""

from __future__ import annotations

from authlib.integrations.starlette_client import OAuth

from mlflow_oidc_auth.config import config
from mlflow_oidc_auth.logger import get_logger

logger = get_logger()

oauth: OAuth = OAuth()
_oidc_client_registered: bool = False


def get_oauth() -> OAuth:
    """Return the module-level OAuth instance."""

    return oauth


def _has_required_config() -> bool:
    """Return True when the minimum OIDC configuration is present.

    A client secret is required unless PKCE is enabled (OIDC_CODE_CHALLENGE=S256).
    """
    if not config.OIDC_CLIENT_ID or not config.OIDC_DISCOVERY_URL:
        return False
    return bool(config.OIDC_CLIENT_SECRET or config.OIDC_CODE_CHALLENGE == "S256")


def _build_scope() -> str:
    """Build the OIDC scope string, adding ``offline_access`` if refresh is enabled.

    Authlib accepts either comma- or space-separated scopes. We normalise to the
    same separator the user configured to keep the rendered authorize URL
    predictable, while making sure ``offline_access`` is present when the
    refresh-token flow is enabled.
    """

    raw = config.OIDC_SCOPE or ""
    if not config.OIDC_USE_REFRESH_TOKEN:
        return raw

    separator = "," if "," in raw else " "
    scopes = [s.strip() for s in raw.replace(",", " ").split() if s.strip()]
    if "offline_access" not in scopes:
        scopes.append("offline_access")
    return separator.join(scopes)


def ensure_oidc_client_registered() -> bool:
    """Ensure the 'oidc' client is registered.

    Returns False if config is incomplete or registration fails.
    """

    global _oidc_client_registered

    if _oidc_client_registered:
        return True

    if not _has_required_config():
        if config.OIDC_CLIENT_ID and config.OIDC_DISCOVERY_URL:
            raise ValueError(
                "OIDC configuration is incomplete: OIDC_CLIENT_SECRET is missing and "
                "OIDC_CODE_CHALLENGE is not set to 'S256'. Provide a client secret or "
                "enable PKCE by setting OIDC_CODE_CHALLENGE=S256."
            )
        return False

    client_kwargs = {"scope": _build_scope()}
    if config.OIDC_CODE_CHALLENGE:
        client_kwargs["code_challenge_method"] = config.OIDC_CODE_CHALLENGE

    if not config.OIDC_SSL_VERIFY:
        logger.warning(
            "OIDC_SSL_VERIFY is disabled: TLS certificate verification for the OIDC "
            "provider is turned off. This is insecure and should only be used in "
            "test environments with self-signed certificates."
        )
        client_kwargs["verify"] = False

    registration_kwargs = {
        "name": "oidc",
        "client_id": config.OIDC_CLIENT_ID,
        "server_metadata_url": config.OIDC_DISCOVERY_URL,
        "client_kwargs": client_kwargs,
    }
    if config.OIDC_CLIENT_SECRET:
        registration_kwargs["client_secret"] = config.OIDC_CLIENT_SECRET

    try:
        oauth.register(**registration_kwargs)
        _oidc_client_registered = True
        return True
    except Exception as exc:
        logger.warning(f"Failed to register OIDC client: {exc}")
        return False


def is_oidc_configured() -> bool:
    """Return True if OIDC config is present and the client is registered."""

    return ensure_oidc_client_registered()


def reset_oauth() -> None:
    """Reset the OAuth instance and registration state (primarily for tests)."""

    global oauth, _oidc_client_registered
    oauth = OAuth()
    _oidc_client_registered = False
