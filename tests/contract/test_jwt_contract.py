"""
Phase 4 — Cross-service JWT contract tests (Python side).

Verifies that tokens produced by Node's generateToken / aiClient._serviceToken
(HS256 fallback) are accepted by Python's require_user middleware, and that the
two sides agree on claim shape, expiry, and key-rollover semantics (RS256).

All tests call require_user() directly (no HTTP round-trip) so the suite runs
without a live server.
"""
import time
import os
import pytest
import jwt
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

# The top-level conftest sets JWT_SECRET via os.environ.setdefault before
# importing the app, so we read whatever was configured there.
TEST_SECRET = os.environ.get("JWT_SECRET", "test-secret-for-kundai-phase2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_user_token(
    role: str = "student",
    user_id: str = "000000000000000000000001",
    student_id=None,
    exp_delta: int = 900,
) -> str:
    """Sign a token with the same logic as Node's generateToken (HS256 fallback)."""
    payload = {
        "id": user_id,
        "role": role,
        "studentId": student_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + exp_delta,
    }
    return jwt.encode(payload, TEST_SECRET, algorithm="HS256")


def _make_service_token() -> str:
    """Sign a token matching aiClient._serviceToken() — { id:'service', role:'service' }."""
    payload = {
        "id": "service",
        "role": "service",
        "iat": int(time.time()),
        "exp": int(time.time()) + 300,  # 5 min
    }
    return jwt.encode(payload, TEST_SECRET, algorithm="HS256")


def _call(token: str) -> dict:
    """Invoke require_user directly, bypassing the HTTP layer."""
    from middleware.auth import require_user
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    return require_user(creds)


# ---------------------------------------------------------------------------
# HS256 contract
# ---------------------------------------------------------------------------

class TestHs256Contract:
    """Tokens signed by Node (HS256 fallback) must be accepted by require_user."""

    @pytest.mark.parametrize("role", ["student", "teacher", "admin", "sys_admin"])
    def test_all_roles_accepted(self, role):
        payload = _call(_make_user_token(role, "usr001"))
        assert payload["role"] == role
        assert payload["id"] == "usr001"

    def test_student_id_claim_preserved(self):
        token = _make_user_token("student", "usr002", student_id="stu42")
        payload = _call(token)
        assert payload.get("studentId") == "stu42"

    def test_null_student_id_preserved(self):
        token = _make_user_token("teacher", "usr003", student_id=None)
        payload = _call(token)
        assert payload.get("studentId") is None

    def test_service_token_accepted(self):
        payload = _call(_make_service_token())
        assert payload["id"] == "service"
        assert payload["role"] == "service"

    def test_exp_claim_present(self):
        payload = _call(_make_user_token())
        assert "exp" in payload
        assert payload["exp"] > int(time.time())

    def test_expired_token_rejected(self):
        token = _make_user_token(exp_delta=-1)
        with pytest.raises(HTTPException) as exc:
            _call(token)
        assert exc.value.status_code == 401

    def test_wrong_secret_rejected(self):
        bad_token = jwt.encode(
            {"id": "x", "role": "student", "exp": int(time.time()) + 900},
            "totally-wrong-secret",
            algorithm="HS256",
        )
        with pytest.raises(HTTPException) as exc:
            _call(bad_token)
        assert exc.value.status_code == 401

    def test_malformed_token_rejected(self):
        with pytest.raises(HTTPException) as exc:
            _call("this.is.not.a.jwt")
        assert exc.value.status_code == 401

    def test_missing_token_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            _call("")
        assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# RS256 contract
# ---------------------------------------------------------------------------

class TestRs256Contract:
    """When JWT_PUBLIC_KEY is set the middleware switches to RS256 verification."""

    @pytest.fixture()
    def rsa_key_pair(self):
        """Generate an inline RSA-2048 key pair for the test."""
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization

        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend(),
        )
        priv_pem = private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        pub_pem = private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()
        return priv_pem, pub_pem

    def test_rs256_token_accepted(self, monkeypatch, rsa_key_pair):
        """Token signed with the private key verifies against the patched public key."""
        import middleware.auth as auth_module

        priv_pem, pub_pem = rsa_key_pair
        monkeypatch.setattr(auth_module, "_verify_key", pub_pem)
        monkeypatch.setattr(auth_module, "_algorithms", ["RS256"])

        token = jwt.encode(
            {"id": "rs-usr", "role": "teacher", "exp": int(time.time()) + 900},
            priv_pem,
            algorithm="RS256",
        )
        payload = _call(token)
        assert payload["id"] == "rs-usr"
        assert payload["role"] == "teacher"

    def test_rs256_wrong_key_rejected(self, monkeypatch, rsa_key_pair):
        """Token signed with key-A is rejected when middleware has key-B as verify key."""
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
        import middleware.auth as auth_module

        priv_a, _ = rsa_key_pair

        # A completely different key for verification
        key_b = rsa.generate_private_key(65537, 2048, default_backend())
        pub_b = key_b.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()

        monkeypatch.setattr(auth_module, "_verify_key", pub_b)
        monkeypatch.setattr(auth_module, "_algorithms", ["RS256"])

        token = jwt.encode(
            {"id": "rs-usr", "role": "student", "exp": int(time.time()) + 900},
            priv_a,
            algorithm="RS256",
        )
        with pytest.raises(HTTPException) as exc:
            _call(token)
        assert exc.value.status_code == 401

    def test_hs256_token_rejected_when_rs256_configured(self, monkeypatch, rsa_key_pair):
        """When the middleware is in RS256 mode an HS256 token must be rejected."""
        import middleware.auth as auth_module

        _, pub_pem = rsa_key_pair
        monkeypatch.setattr(auth_module, "_verify_key", pub_pem)
        monkeypatch.setattr(auth_module, "_algorithms", ["RS256"])

        hs_token = _make_user_token("student")
        with pytest.raises(HTTPException) as exc:
            _call(hs_token)
        assert exc.value.status_code == 401
