import base64
import hashlib
import os
import re
import sys
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from modelpicker.server import create_server


async def test_stdio_client_can_import_inspect_and_delete(settings):
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "modelpicker", "serve"],
        env={
            **os.environ,
            "MODELPICKER_HOME": str(settings.data_dir),
            "MODELPICKER_DATASET_DIR": str(settings.dataset_dir),
            "MODELPICKER_CACHE_DIR": str(settings.cache_dir),
            "TRANSPORT": "stdio",
        },
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert len(tools.tools) == 23
            capabilities = await session.call_tool("get_capabilities", {})
            assert not capabilities.isError
            imported = await session.call_tool(
                "import_dataset_rows",
                {"rows": [{"text": "great", "label": "positive"}, {"text": "bad", "label": "negative"}]},
            )
            assert not imported.isError
            dataset_id = imported.structuredContent["id"]
            inspected = await session.call_tool("inspect_dataset", {"dataset_id": dataset_id})
            assert inspected.structuredContent["row_count"] == 2
            assert "preview" not in inspected.structuredContent
            rejected = await session.call_tool("import_dataset_file", {"path": "/etc/passwd"})
            assert rejected.isError
            deleted = await session.call_tool("delete_saved", {"kind": "dataset", "record_id": dataset_id})
            assert not deleted.isError


def test_remote_cannot_start_unauthenticated(settings):
    with pytest.raises(ValueError, match="never exposed"):
        create_server(settings, transport="streamable-http")
    with pytest.raises(ValueError, match="HTTPS origin"):
        create_server(
            settings, transport="streamable-http", public_url="http://example.com", owner_password="x" * 32
        )


async def test_remote_oauth_pkce_and_persistent_http_tools(settings):
    password = "owner-secret-" + "a" * 32
    server = create_server(
        settings, transport="streamable-http", public_url="https://demo.example", owner_password=password
    )
    app = server.streamable_http_app()
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://demo.example"
        ) as client:
            unauthorized = await client.post("/mcp", json={})
            assert unauthorized.status_code == 401
            metadata = await client.get("/.well-known/oauth-protected-resource/mcp")
            assert metadata.status_code == 200
            registration = await client.post(
                "/register",
                json={
                    "client_name": "Test chat",
                    "redirect_uris": ["https://client.example/callback"],
                    "token_endpoint_auth_method": "none",
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                    "scope": "modelpicker",
                },
            )
            assert registration.status_code == 201, registration.text
            client_id = registration.json()["client_id"]
            verifier = "verifier-" + "a" * 64
            challenge = (
                base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
            )
            authorization = await client.get(
                "/authorize",
                params={
                    "client_id": client_id,
                    "response_type": "code",
                    "redirect_uri": "https://client.example/callback",
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "state": "state-test",
                    "scope": "modelpicker",
                    "resource": "https://demo.example/mcp",
                },
            )
            assert authorization.status_code in {302, 303, 307}, authorization.text
            consent = await client.get(authorization.headers["location"])
            assert consent.status_code == 200
            assert "Test chat" in consent.text
            request_id = parse_qs(urlsplit(authorization.headers["location"]).query)["request_id"][0]
            csrf = re.search(r'name="csrf" value="([^"]+)"', consent.text)[1]
            wrong = await client.post(
                "/owner/approve", data={"request_id": request_id, "csrf": csrf, "password": "wrong"}
            )
            assert wrong.status_code == 403
            approved = await client.post(
                "/owner/approve", data={"request_id": request_id, "csrf": csrf, "password": password}
            )
            assert approved.status_code == 303, approved.text
            query = parse_qs(urlsplit(approved.headers["location"]).query)
            assert query["state"] == ["state-test"]
            exchange = {
                "grant_type": "authorization_code",
                "code": query["code"][0],
                "client_id": client_id,
                "redirect_uri": "https://client.example/callback",
                "code_verifier": verifier,
                "resource": "https://demo.example/mcp",
            }
            token_response = await client.post("/token", data=exchange)
            assert token_response.status_code == 200, token_response.text
            replay = await client.post("/token", data=exchange)
            assert replay.status_code == 400
            headers = {
                "Authorization": "Bearer " + token_response.json()["access_token"],
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2025-06-18",
            }

            async def call(name, arguments):
                response = await client.post(
                    "/mcp",
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {"name": name, "arguments": arguments},
                    },
                )
                assert response.status_code == 200, response.text
                return response.json()["result"]

            imported = await call(
                "import_dataset_rows",
                {"rows": [{"text": "good", "label": "positive"}, {"text": "bad", "label": "negative"}]},
            )
            assert not imported.get("isError"), imported
            dataset_id = imported["structuredContent"]["id"]
            # Separate stateless requests must use the same persistent Engine.
            inspected = await call("inspect_dataset", {"dataset_id": dataset_id})
            assert inspected["structuredContent"]["row_count"] == 2
            wrong_host = await client.post("/mcp", headers={**headers, "Host": "evil.example"}, json={})
            assert wrong_host.status_code == 421
            refreshed = await client.post(
                "/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "refresh_token": token_response.json()["refresh_token"],
                    "scope": "modelpicker",
                },
            )
            assert refreshed.status_code == 200, refreshed.text
            old_access = await client.post("/mcp", headers=headers, json={})
            assert old_access.status_code == 401
            replay_refresh = await client.post(
                "/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "refresh_token": token_response.json()["refresh_token"],
                    "scope": "modelpicker",
                },
            )
            assert replay_refresh.status_code == 400
            revoked = await client.post(
                "/revoke",
                data={
                    "token": refreshed.json()["refresh_token"],
                    "client_id": client_id,
                    "token_type_hint": "refresh_token",
                },
            )
            assert revoked.status_code == 200, revoked.text
            new_access = await client.post(
                "/mcp",
                headers={**headers, "Authorization": "Bearer " + refreshed.json()["access_token"]},
                json={},
            )
            assert new_access.status_code == 401
