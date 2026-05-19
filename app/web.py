# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Web UI — lightweight FastAPI server with WebSocket broadcasting.

Provides a Broadcaster class for thread-safe message delivery from the
pipeline thread to all connected browser clients, and a FastAPI app
that serves the single-page frontend and a WebSocket endpoint.
"""

import asyncio
import json
import threading
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

STATIC_DIR = Path(__file__).parent.parent / "static"


class Broadcaster:
    """Thread-safe fan-out from the synchronous pipeline to async WebSocket clients.

    Each connected client gets its own asyncio.Queue.  The pipeline calls
    send() from any thread; messages are routed into the event loop via
    call_soon_threadsafe.

    Also holds shared push-to-talk (PTT) state: cleared = muted (default),
    set = unmuted / listening.  Any client can toggle via WebSocket.
    """

    def __init__(self):
        self._clients: dict[asyncio.Queue, None] = {}
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ptt = threading.Event()
        # Cached one-shot init payload (models, platform, tools, config) sent
        # to each new WebSocket on connect. Without this, the pipeline's
        # initial broadcast lands before any browser is connected and gets
        # dropped, leaving the Tools panel and config block empty.
        self._info: Optional[dict] = None
        # Latest camera-list payload (local-media mode), cached like _info
        # so a freshly-connected browser can populate its dropdown.
        self._camera_list: Optional[dict] = None
        self._camera_power: Optional[dict] = None
        # Callback(int) -> bool that hot-swaps the active camera. Set by
        # the entry point only in local-media mode; None otherwise.
        self._camera_switch: Optional[Callable[[int], bool]] = None

    def set_camera_switch(self, fn: Optional[Callable[[int], bool]]):
        self._camera_switch = fn

    @property
    def camera_switch(self) -> Optional[Callable[[int], bool]]:
        return self._camera_switch

    def cached_camera_list(self) -> Optional[dict]:
        with self._lock:
            return self._camera_list

    def cached_camera_power(self) -> Optional[dict]:
        with self._lock:
            return self._camera_power

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def register(self, q: asyncio.Queue):
        with self._lock:
            self._clients[q] = None

    def unregister(self, q: asyncio.Queue):
        with self._lock:
            self._clients.pop(q, None)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    @property
    def ptt_active(self) -> bool:
        return self._ptt.is_set()

    def set_ptt(self, active: bool):
        if active:
            self._ptt.set()
        else:
            self._ptt.clear()
        self.send({"type": "ptt_state", "active": active})

    def send(self, msg: dict):
        """Enqueue *msg* for every connected client (thread-safe)."""
        # Cache `info` so late-arriving clients still get registry/config
        # state. Subsequent info messages replace the cache.
        if msg.get("type") == "info":
            with self._lock:
                self._info = msg
        elif msg.get("type") == "camera_list":
            with self._lock:
                self._camera_list = msg
        elif msg.get("type") == "camera_power":
            with self._lock:
                self._camera_power = msg
        loop = self._loop
        if not loop:
            return
        with self._lock:
            for q in self._clients:
                try:
                    loop.call_soon_threadsafe(q.put_nowait, msg)
                except asyncio.QueueFull:
                    pass
                except Exception:
                    pass

    def cached_info(self) -> Optional[dict]:
        with self._lock:
            return self._info


def create_app(broadcaster: Broadcaster) -> FastAPI:
    app = FastAPI(title="Reachy Mini Vision Chat")

    @app.on_event("startup")
    async def _startup():
        broadcaster.set_loop(asyncio.get_event_loop())

    @app.get("/")
    async def _index():
        path = STATIC_DIR / "index.html"
        if path.exists():
            return FileResponse(
                path, media_type="text/html",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )
        return HTMLResponse("<h1>static/index.html not found</h1>", status_code=404)

    @app.websocket("/ws")
    async def _ws(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=128)
        broadcaster.register(q)

        q.put_nowait({"type": "ptt_state", "active": broadcaster.ptt_active})
        cached = broadcaster.cached_info()
        if cached is not None:
            q.put_nowait(cached)
        cached_cams = broadcaster.cached_camera_list()
        if cached_cams is not None:
            q.put_nowait(cached_cams)
        cached_cp = broadcaster.cached_camera_power()
        if cached_cp is not None:
            q.put_nowait(cached_cp)

        async def _sender():
            try:
                while True:
                    msg = await q.get()
                    await ws.send_text(json.dumps(msg))
            except (WebSocketDisconnect, Exception):
                pass

        async def _receiver():
            try:
                while True:
                    data = await ws.receive_text()
                    try:
                        msg = json.loads(data)
                        if msg.get("type") == "ptt":
                            broadcaster.set_ptt(bool(msg.get("active", False)))
                        elif msg.get("type") == "set_camera":
                            fn = broadcaster.camera_switch
                            if fn is not None:
                                # cv2.VideoCapture open can block ~1s;
                                # keep it off the event loop.
                                asyncio.get_event_loop().run_in_executor(
                                    None, fn, int(msg.get("index", 0))
                                )
                    except (ValueError, KeyError):
                        pass
            except (WebSocketDisconnect, Exception):
                pass

        send_task = asyncio.ensure_future(_sender())
        recv_task = asyncio.ensure_future(_receiver())
        try:
            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
        except Exception:
            send_task.cancel()
            recv_task.cancel()
        finally:
            broadcaster.unregister(q)

    return app


def start_web_server(
    broadcaster: Broadcaster,
    host: str = "0.0.0.0",
    port: int = 8090,
) -> threading.Thread:
    """Start uvicorn in a daemon thread.  Returns immediately."""
    import uvicorn

    app = create_app(broadcaster)

    def _run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    t = threading.Thread(target=_run, daemon=True, name="web-server")
    t.start()
    return t
