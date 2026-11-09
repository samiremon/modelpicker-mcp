"""Single-owner OAuth for a personal remote MCP worker; not a multi-tenant identity service."""

from __future__ import annotations

import hashlib
import html
import secrets
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from mcp.server.auth.middleware.client_auth import AuthenticationError, ClientAuthenticator
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizeError,
    RefreshToken,
    RegistrationError,
    TokenError,
)
from mcp.shared.auth import OAuthToken
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response


class OwnerOAuth:
    def __init__(self, public_url: str, password: str, resource_path: str = "/mcp"):
        if len(password) < 24:
            raise ValueError(
                "MODELPICKER_OWNER_PASSWORD must have at least 24 characters; generate it with modelpicker setup"
            )
        self.public_url = public_url.rstrip("/")
        self.resource = self.public_url + resource_path
        self.password = password
        self.clients, self.pending, self.codes, self.tokens = {}, {}, {}, {}
        self.refresh_tokens, self.pairs = {}, {}
        self.attempts = []

    def prune(self):
        current = time.time()
        self.pending = {k: v for k, v in self.pending.items() if v["expires"] > current}
        self.codes = {k: v for k, v in self.codes.items() if v.expires_at > current}
        self.tokens = {k: v for k, v in self.tokens.items() if v.expires_at > current}
        self.refresh_tokens = {k: v for k, v in self.refresh_tokens.items() if v.expires_at > current}
        live = self.tokens.keys() | self.refresh_tokens.keys()
        self.pairs = {k: v for k, v in self.pairs.items() if k in live and v in live}
        self.attempts = [t for t in self.attempts if current - t < 60]

    async def get_client(self, client_id):
        return self.clients.get(client_id)

    async def register_client(self, client_info):
        if len(self.clients) >= 100:
            raise RegistrationError(
                "invalid_client_metadata", "Client registration limit reached; restart service"
            )
        self.clients[client_info.client_id] = client_info

    async def authorize(self, client, params):
        self.prune()
        if len(self.pending) >= 100:
            raise AuthorizeError("temporarily_unavailable", "Too many pending authorizations")
        if params.resource and params.resource.rstrip("/") != self.resource:
            raise AuthorizeError("invalid_request", "Resource does not match this server")
        request_id = secrets.token_urlsafe(32)
        self.pending[request_id] = {"client": client, "params": params, "expires": time.time() + 600}
        return f"{self.public_url}/owner/approve?request_id={request_id}"

    async def approval(self, request: Request):
        self.prune()
        headers = {
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "default-src 'none'; form-action 'self'; frame-ancestors 'none'",
            "X-Content-Type-Options": "nosniff",
        }
        if request.method == "GET":
            request_id = request.query_params.get("request_id", "")
            pending = self.pending.get(request_id)
            if not pending:
                return HTMLResponse(
                    "Authorization expired. Reconnect from your chat client.",
                    status_code=400,
                    headers=headers,
                )
            csrf = secrets.token_urlsafe(32)
            pending["csrf"] = csrf
            client_name = html.escape(pending["client"].client_name or "MCP client")
            body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Connect ModelPicker</title>
<h1>Connect {client_name} to your local ModelPicker</h1>
<p>This grants access to your saved evaluations, authorized dataset directory, and CPU evaluation tools.</p>
<p>Only continue if you initiated this connection. Enter your local ModelPicker owner password.</p>
<form method="post"><input type="hidden" name="request_id" value="{html.escape(request_id)}">
<input type="hidden" name="csrf" value="{csrf}">
<label>Owner password <input type="password" name="password" required autocomplete="current-password"></label>
<button type="submit">Authorize connection</button></form></html>'''
            response = HTMLResponse(body, headers=headers)
            response.set_cookie(
                "modelpicker_csrf",
                csrf,
                secure=self.public_url.startswith("https:"),
                httponly=True,
                samesite="strict",
                max_age=600,
                path="/owner/approve",
            )
            return response
        if len(self.attempts) >= 10:
            return HTMLResponse("Too many attempts. Wait one minute.", status_code=429, headers=headers)
        self.attempts.append(time.time())
        form = await request.form()
        request_id = str(form.get("request_id", ""))
        pending = self.pending.get(request_id)
        csrf = str(form.get("csrf", ""))
        if (
            not pending
            or not csrf
            or not secrets.compare_digest(csrf, pending.get("csrf", ""))
            or not secrets.compare_digest(csrf, request.cookies.get("modelpicker_csrf", ""))
            or not secrets.compare_digest(str(form.get("password", "")).encode(), self.password.encode())
        ):
            return HTMLResponse(
                "Authorization failed. Check your password or reconnect.", status_code=403, headers=headers
            )
        self.pending.pop(request_id)
        params, client = pending["params"], pending["client"]
        code = AuthorizationCode(
            code=secrets.token_urlsafe(32),
            client_id=client.client_id,
            scopes=params.scopes or ["modelpicker"],
            expires_at=time.time() + 120,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=self.resource,
        )
        self.codes[code.code] = code
        parts = urlsplit(str(params.redirect_uri))
        query = parse_qsl(parts.query) + [("code", code.code)]
        if params.state is not None:
            query.append(("state", params.state))
        response = RedirectResponse(
            urlunsplit(parts._replace(query=urlencode(query))), status_code=303, headers=headers
        )
        response.delete_cookie("modelpicker_csrf", path="/owner/approve")
        return response

    async def load_authorization_code(self, client, authorization_code):
        self.prune()
        code = self.codes.get(authorization_code)
        return code if code and code.client_id == client.client_id else None

    async def exchange_authorization_code(self, client, authorization_code):
        code = self.codes.pop(authorization_code.code, None)
        if not code or code.client_id != client.client_id or code.expires_at <= time.time():
            raise TokenError("invalid_grant", "Expired or used authorization code")
        return self.issue_tokens(client.client_id, code.scopes)

    def issue_tokens(self, client_id, scopes):
        token, refresh = secrets.token_urlsafe(48), secrets.token_urlsafe(48)
        access_hash = hashlib.sha256(token.encode()).hexdigest()
        refresh_hash = hashlib.sha256(refresh.encode()).hexdigest()
        self.tokens[access_hash] = AccessToken(
            token=token,
            client_id=client_id,
            scopes=scopes,
            expires_at=int(time.time()) + 8 * 3600,
            resource=self.resource,
        )
        self.refresh_tokens[refresh_hash] = RefreshToken(
            token=refresh,
            client_id=client_id,
            scopes=scopes,
            expires_at=int(time.time()) + 30 * 86400,
            resource=self.resource,
        )
        self.pairs[access_hash], self.pairs[refresh_hash] = refresh_hash, access_hash
        return OAuthToken(
            access_token=token, refresh_token=refresh, expires_in=8 * 3600, scope=" ".join(scopes)
        )

    async def load_access_token(self, token):
        self.prune()
        return self.tokens.get(hashlib.sha256(token.encode()).hexdigest())

    async def load_refresh_token(self, client, refresh_token):
        self.prune()
        token = self.refresh_tokens.get(hashlib.sha256(refresh_token.encode()).hexdigest())
        return token if token and token.client_id == client.client_id else None

    async def exchange_refresh_token(self, client, refresh_token, scopes):
        token = await self.load_refresh_token(client, refresh_token.token)
        if not token or not set(scopes).issubset(token.scopes):
            raise TokenError("invalid_grant", "Invalid refresh token or scopes")
        await self.revoke_token(token)
        return self.issue_tokens(client.client_id, scopes)

    async def revoke_token(self, token):
        key = hashlib.sha256(token.token.encode()).hexdigest()
        partner = self.pairs.pop(key, None)
        for item in (key, partner):
            if item:
                self.tokens.pop(item, None)
                self.refresh_tokens.pop(item, None)
                self.pairs.pop(item, None)

    async def revocation(self, request: Request):
        """RFC 7009 endpoint: public PKCE clients must not be required to send a client_secret."""
        headers = {"Cache-Control": "no-store", "Pragma": "no-cache"}
        try:
            client = await ClientAuthenticator(self).authenticate_request(request)
        except AuthenticationError:
            return JSONResponse({"error": "invalid_client"}, status_code=401, headers=headers)
        form = await request.form()
        value = form.get("token")
        if not isinstance(value, str) or not value or len(value) > 4096:
            return JSONResponse({"error": "invalid_request"}, status_code=400, headers=headers)
        token = await self.load_access_token(value) or await self.load_refresh_token(client, value)
        if token and token.client_id == client.client_id:
            await self.revoke_token(token)
        return Response(status_code=200, headers=headers)
