import os
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

_bearer = HTTPBearer()

# Prefer RS256 (public key only — private key stays in the Node backend).
# Fall back to HS256 during key-rollout or in development.
_raw_public_key = os.getenv("JWT_PUBLIC_KEY", "").replace("\\n", "\n").strip()
if _raw_public_key:
    _verify_key  = _raw_public_key
    _algorithms  = ["RS256"]
else:
    _verify_key  = os.getenv("JWT_SECRET", "")
    _algorithms  = ["HS256"]


def require_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    if not _verify_key:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Auth not configured")
    try:
        payload = jwt.decode(credentials.credentials, _verify_key, algorithms=_algorithms)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return payload
