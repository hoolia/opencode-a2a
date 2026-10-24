from __future__ import annotations

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from opencode_a2a.jwt_auth import TrustedIssuer, authenticate_tenant_jwt

KC = "https://login.example/realms/public"
K8S = "https://kubernetes.default.svc"


class _StaticKey:
    def __init__(self, key):  # noqa: ANN001
        self.key = key


class _FakeJWKClient:
    def __init__(self, public_key):  # noqa: ANN001
        self._key = public_key

    def get_signing_key_from_jwt(self, token: str) -> _StaticKey:  # noqa: ARG002
        return _StaticKey(self._key)


@pytest.fixture(scope="module")
def keys():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private, other


def _issuers(public_key, audiences=()):  # noqa: ANN001
    client = _FakeJWKClient(public_key)
    return (
        TrustedIssuer(issuer=KC, kind="oidc", audiences=tuple(audiences), client=client),
        TrustedIssuer(issuer=K8S, kind="kubernetes", audiences=(), client=client),
    )


def _token(key, **claims):  # noqa: ANN001
    body = {"exp": int(time.time()) + 300, **claims}
    return jwt.encode(body, key, algorithm="RS256")


def test_realm_user_in_tenant_group(keys):
    private, _ = keys
    token = _token(private, iss=KC, groups=["grncloud-admins"], email="sam@grn.cloud")
    principal = authenticate_tenant_jwt(
        issuers=_issuers(private.public_key()), tenant="grncloud", token=token
    )
    assert principal is not None
    assert principal.identity == "sam@grn.cloud"
    assert principal.capabilities == ()


def test_realm_user_of_other_tenant_refused(keys):
    private, _ = keys
    token = _token(private, iss=KC, groups=["hoolia-admins", "grncloudx-admins"], sub="u1")
    assert (
        authenticate_tenant_jwt(
            issuers=_issuers(private.public_key()), tenant="grncloud", token=token
        )
        is None
    )


def test_serviceaccount_of_tenant_namespace(keys):
    private, _ = keys
    claims = {"namespace": "grncloud-kagent", "serviceaccount": {"name": "kagent-controller"}}
    token = _token(private, iss=K8S, **{"kubernetes.io": claims})
    principal = authenticate_tenant_jwt(
        issuers=_issuers(private.public_key()), tenant="grncloud", token=token
    )
    assert principal is not None
    assert principal.identity == "system:serviceaccount:grncloud-kagent:kagent-controller"


def test_serviceaccount_of_other_tenant_refused(keys):
    private, _ = keys
    token = _token(private, iss=K8S, **{"kubernetes.io": {"namespace": "hoolia-aib"}})
    assert (
        authenticate_tenant_jwt(
            issuers=_issuers(private.public_key()), tenant="grncloud", token=token
        )
        is None
    )


def test_bad_signature_expired_and_unknown_issuer_refused(keys):
    private, other = keys
    issuers = _issuers(private.public_key())
    forged = _token(other, iss=KC, groups=["grncloud-admins"])
    expired = jwt.encode(
        {"iss": KC, "groups": ["grncloud"], "exp": int(time.time()) - 10},
        private,
        algorithm="RS256",
    )
    unknown = _token(private, iss="https://evil.example", groups=["grncloud"])
    for token in (forged, expired, unknown, "not-a-jwt"):
        assert authenticate_tenant_jwt(issuers=issuers, tenant="grncloud", token=token) is None


def test_audience_enforced_when_listed(keys):
    private, _ = keys
    issuers = _issuers(private.public_key(), audiences=["agents.grn.cloud"])
    good = _token(private, iss=KC, aud="agents.grn.cloud", groups=["grncloud"])
    bad = _token(private, iss=KC, aud="other", groups=["grncloud"])
    assert authenticate_tenant_jwt(issuers=issuers, tenant="grncloud", token=good) is not None
    assert authenticate_tenant_jwt(issuers=issuers, tenant="grncloud", token=bad) is None
