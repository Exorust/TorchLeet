"""Single-port entrypoint for the Space.

A HuggingFace Space exposes exactly one port, so both surfaces live behind this
FastAPI app:

    /        the Gradio catalogue (mounted)
    /lab/*   JupyterLab, reverse-proxied from 127.0.0.1:8888

JupyterLab needs WebSockets for kernel traffic, not just HTTP, so both are
forwarded. If the WebSocket path is broken the lab loads but cells never run,
which is the failure mode to watch for.
"""
from __future__ import annotations

import asyncio
import os

import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from starlette.websockets import WebSocketDisconnect

import gradio as gr

from app import CSS, THEME, demo

LAB_HOST = os.environ.get("LAB_HOST", "127.0.0.1")
LAB_PORT = int(os.environ.get("LAB_PORT", 8888))
LAB_HTTP = f"http://{LAB_HOST}:{LAB_PORT}"
LAB_WS = f"ws://{LAB_HOST}:{LAB_PORT}"

# Hop-by-hop headers must not be forwarded (RFC 7230); passing content-length or
# transfer-encoding through a proxy that re-frames the body corrupts responses.
DROP = {"content-length", "transfer-encoding", "connection", "keep-alive",
        "upgrade", "host"}

api = FastAPI()
client = httpx.AsyncClient(base_url=LAB_HTTP, timeout=None, follow_redirects=False)


# With --ServerApp.base_url=/lab, JupyterLab itself is served at /lab/lab, so the
# real notebook URL is /lab/lab/tree/<path>. These redirects keep the link we hand
# out ( /lab and /lab/tree/<path> ) readable. Both must be declared before the
# catch-all below, since FastAPI matches routes in registration order.
@api.get("/lab")
async def lab_root():
    return RedirectResponse("/lab/lab/tree")


@api.get("/lab/tree/{path:path}")
async def lab_tree(path: str):
    return RedirectResponse(f"/lab/lab/tree/{path}")


@api.websocket("/lab/{path:path}")
async def lab_ws(ws: WebSocket, path: str):
    """Bidirectional passthrough for kernel/terminal sockets."""
    await ws.accept(subprotocol=ws.headers.get("sec-websocket-protocol"))
    qs = ws.url.query
    target = f"{LAB_WS}/lab/{path}" + (f"?{qs}" if qs else "")
    try:
        async with websockets.connect(target, open_timeout=20,
                                      max_size=None, ping_interval=None) as up:
            async def to_upstream():
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect
                    if (data := msg.get("bytes")) is not None:
                        await up.send(data)
                    elif (text := msg.get("text")) is not None:
                        await up.send(text)

            async def to_client():
                async for msg in up:
                    if isinstance(msg, bytes):
                        await ws.send_bytes(msg)
                    else:
                        await ws.send_text(msg)

            done, pending = await asyncio.wait(
                {asyncio.create_task(to_upstream()), asyncio.create_task(to_client())},
                return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
    except (WebSocketDisconnect, ConnectionError, OSError):
        pass
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass


@api.api_route("/lab/{path:path}",
               methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def lab_http(request: Request, path: str):
    url = f"/lab/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP}
    req = client.build_request(request.method, url, headers=headers,
                               content=request.stream())
    up = await client.send(req, stream=True)
    out = {k: v for k, v in up.headers.items() if k.lower() not in DROP}
    return StreamingResponse(up.aiter_raw(), status_code=up.status_code,
                             headers=out, background=up.aclose)


@api.get("/healthz")
async def healthz():
    return {"ok": True}


api = gr.mount_gradio_app(api, demo, path="/", theme=THEME, css=CSS)
