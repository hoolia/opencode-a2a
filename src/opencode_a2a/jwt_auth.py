"""Bearer JWTs from trusted issuers, admitted when the caller belongs to the tenant.

A caller belongs to tenant ``t`` when a Kubernetes ServiceAccount token names a
namespace ``t`` or ``t-*``, or an OIDC token carries a ``groups`` value ``t`` or
``t-*``. Signatures are checked against each issuer's JWKS, cached for five
minutes; audiences are checked only when an issuer lists them.
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jwt

from .auth import AuthenticatedPrincipal
from .config import Settings

_JWKS_LIFESPAN_SECONDS = 300
_ALGORITHMS = ["RS256", "RS384", "RS512", "ES256", "ES384", "PS256"]


@dataclass(frozen=True)
class TrustedIssuer:
    issuer: str
    kind: str
    audiences: tuple[str, ...]
    client: jwt.PyJWKClient


def _member(value: str, tenant: str) -> bool:
    return value == tenant or value.startswith(tenant + "-")


class _TokenJWKClient(jwt.PyJWKClient):
    """Sends the pod's ServiceAccount token, re-read on every fetch since it rotates."""

    def __init__(self, uri: str, token_file: str, **kwargs: Any) -> None:
        super().__init__(uri, **kwargs)
        self._token_file = token_file

    def fetch_data(self) -> Any:
        self.headers = {"Authorization": "Bearer " + Path(self._token_file).read_text().strip()}
        return super().fetch_data()


def _jwks_client(jwks_uri: str, ca_file: str | None, token_file: str | None) -> jwt.PyJWKClient:
    context = ssl.create_default_context(cafile=ca_file) if ca_file else None
    kwargs: dict[str, Any] = {
        "cache_jwk_set": True,
        "lifespan": _JWKS_LIFESPAN_SECONDS,
        "ssl_context": context,
    }
    if token_file:
        return _TokenJWKClient(jwks_uri, token_file, **kwargs)
    return jwt.PyJWKClient(jwks_uri, **kwargs)


def build_trusted_issuers(settings: Settings) -> tuple[TrustedIssuer, ...]:
    if not settings.a2a_tenant:
        return ()
    issuers = []
    for entry in settings.a2a_jwt_issuers:
        jwks_uri = entry.jwks_uri or entry.issuer.rstrip("/") + (
            "/openid/v1/jwks" if entry.kind == "kubernetes" else "/protocol/openid-connect/certs"
        )
        issuers.append(
            TrustedIssuer(
                issuer=entry.issuer,
                kind=entry.kind,
                audiences=tuple(entry.audiences or ()),
                client=_jwks_client(jwks_uri, entry.ca_file, entry.token_file),
            )
        )
    return tuple(issuers)


def _tenant_identity(kind: str, claims: dict[str, Any], tenant: str) -> str | None:
    if kind == "kubernetes":
        k8s = claims.get("kubernetes.io") or {}
        namespace = k8s.get("namespace") or ""
        account = (k8s.get("serviceaccount") or {}).get("name") or ""
        if _member(namespace, tenant):
            return f"system:serviceaccount:{namespace}:{account}"
        return None
    groups = claims.get("groups") or []
    if isinstance(groups, str):
        groups = [groups]
    if any(_member(str(g).lstrip("/"), tenant) for g in groups):
        return str(claims.get("email") or claims.get("preferred_username") or claims.get("sub"))
    return None


def authenticate_tenant_jwt(
    *, issuers: tuple[TrustedIssuer, ...], tenant: str, token: str
) -> AuthenticatedPrincipal | None:
    """Blocking (JWKS fetch on a cache miss); call it from a worker thread."""
    if not issuers or token.count(".") != 2:
        return None
    try:
        unverified_iss = jwt.decode(token, options={"verify_signature": False}).get("iss")
    except jwt.PyJWTError:
        return None
    for issuer in issuers:
        if issuer.issuer != unverified_iss:
            continue
        try:
            key = issuer.client.get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token,
                key,
                algorithms=_ALGORITHMS,
                issuer=issuer.issuer,
                audience=list(issuer.audiences) or None,
                options={"verify_aud": bool(issuer.audiences), "require": ["exp", "iss"]},
            )
        except jwt.PyJWTError:
            return None
        identity = _tenant_identity(issuer.kind, claims, tenant)
        if identity is None:
            return None
        return AuthenticatedPrincipal(identity=identity, auth_scheme="bearer")
    return None
