import hmac

from fastapi import HTTPException, Request, status

from .settings import Settings

_BEARER_PREFIX = "Bearer "


class RequestGuard:
    def __init__(self, settings: Settings) -> None:
        self._token = settings.token
        self._allowed_origins = set(settings.effective_allowed_origins)
        self._allowed_hosts = settings.allowed_hosts

    def __call__(self, request: Request) -> None:
        self._check_host(request.headers.get("host", ""))
        self._check_origin(request.headers.get("origin"))
        self._check_token(request.headers.get("authorization", ""))

    def _check_host(self, host: str) -> None:
        if host not in self._allowed_hosts:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Host is not allowed")

    def _check_origin(self, origin: str | None) -> None:
        if origin is not None and origin not in self._allowed_origins:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Origin is not allowed")

    def _check_token(self, authorization: str) -> None:
        supplied = authorization.removeprefix(_BEARER_PREFIX) if authorization.startswith(_BEARER_PREFIX) else ""
        if not supplied or not hmac.compare_digest(supplied.encode(), self._token.encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing bearer token")
