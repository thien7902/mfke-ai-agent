"""Health check HTTP server for liveness/readiness probes."""
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Optional

from aiohttp import web

from src.bot.utils.config import config

logger = logging.getLogger(__name__)

# Loopback-only addresses — the snapshot tool inside the container
# reaches localhost:8080/kb_match, so any non-loopback request is
# refused. Defense-in-depth; aiohttp does not authenticate clients.
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@dataclass
class HealthStatus:
    """Health status of the bot and its dependencies."""
    bot_alive: bool = True
    mongodb_connected: bool = True
    holmes_ready: bool = True

    @property
    def is_healthy(self) -> bool:
        return self.bot_alive and self.mongodb_connected and self.holmes_ready

    @property
    def is_ready(self) -> bool:
        return self.bot_alive and self.mongodb_connected


class HealthChecker:
    """Manages health status and provides HTTP endpoints."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.status = HealthStatus()
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        # KB integration: lazily set by main.py. None means kb_rules
        # collection is not in use — /kb_match returns empty block.
        self._kb_store = None

    def create_app(self) -> web.Application:
        """Create the aiohttp application with health endpoints."""
        # client_max_size raised to 4 MB so the snapshot tool can POST a
        # full gardener_controlplane_snapshot body (~700k chars Vietnamese
        # → 1.5–2 MB UTF-8). Past this size aiohttp returns 413 before
        # kb_match_handler even runs, and the snapshot tool sees an empty
        # KB block — silently broken.
        app = web.Application(client_max_size=4 * 1024 * 1024)
        app.router.add_get("/health", self.health_handler)
        app.router.add_get("/live", self.liveness_handler)
        app.router.add_get("/ready", self.readiness_handler)
        # KB rule retrieval for the snapshot tool. Body = raw snapshot
        # signals (events + failed pods + shoot conditions). Returns the
        # top-K distilled if-then rules block to append to the snapshot
        # tool's stdout before Holmes sees the next LLM round. Loopback
        # only — see kb_match_handler.
        app.router.add_post("/kb_match", self.kb_match_handler)
        app.router.add_get("/kb_stats", self.kb_stats_handler)
        return app

    async def health_handler(self, request: web.Request) -> web.Response:
        """Full health check endpoint - checks all dependencies."""
        checks = {
            "bot": self.status.bot_alive,
            "mongodb": self.status.mongodb_connected,
            "holmes": self.status.holmes_ready,
        }

        status_code = 200 if self.status.is_healthy else 503
        return web.json_response({
            "status": "healthy" if self.status.is_healthy else "unhealthy",
            "checks": checks,
        }, status=status_code)

    async def liveness_handler(self, request: web.Request) -> web.Response:
        """Liveness probe - only checks if the bot process is alive."""
        # Return 503 when the bot is not alive so K8s/Docker restarts the
        # container. Previously this always returned 200, making the probe
        # useless as a restart trigger.
        status_code = 200 if self.status.bot_alive else 503
        return web.json_response({
            "status": "alive" if self.status.bot_alive else "not_alive",
            "bot": self.status.bot_alive,
        }, status=status_code)

    async def readiness_handler(self, request: web.Request) -> web.Response:
        """Readiness probe - checks if bot can handle requests."""
        # Ready when bot and MongoDB are available
        checks = {
            "bot": self.status.bot_alive,
            "mongodb": self.status.mongodb_connected,
        }

        status_code = 200 if self.status.is_ready else 503
        return web.json_response({
            "status": "ready" if self.status.is_ready else "not_ready",
            "checks": checks,
        }, status=status_code)

    async def start(self):
        """Start the HTTP health check server."""
        self._app = self.create_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        logger.info(f"Health check server started on http://{self.host}:{self.port}")

    async def stop(self):
        """Stop the HTTP health check server."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        logger.info("Health check server stopped")

    def set_bot_alive(self, alive: bool):
        """Update bot alive status."""
        self.status.bot_alive = alive

    def set_mongodb_connected(self, connected: bool):
        """Update MongoDB connection status."""
        self.status.mongodb_connected = connected

    def set_holmes_ready(self, ready: bool):
        """Update Holmes service readiness."""
        self.status.holmes_ready = ready

    def set_kb_store(self, store) -> None:
        """Inject the KBStore. Called from main.py wiring (ungated — even
        when KB_ENABLED=false, the store exists; the handler short-circuits
        to empty block when disabled).
        """
        self._kb_store = store

    async def kb_match_handler(self, request: web.Request) -> web.Response:
        """POST /kb_match — body = raw snapshot text. Returns the top-K
        distilled if-then rules block to append to snapshot tool output.

        Loopback-only. Fire-and-forget increments match_count AFTER the
        response is sent (via asyncio.ensure_future) so a slow Mongo write
        never blocks the snapshot tool's curl. Failures of any kind →
        empty KB block + 200 (the snapshot must never break because KB
        is unavailable).
        """
        if request.remote not in _LOOPBACK:
            return web.Response(status=403, text="forbidden")
        body = await request.text()
        if not body or not body.strip():
            return web.Response(status=400, text="empty body")
        try:
            from src.bot.services.kb_store import format_rules_block
            if self._kb_store is None:
                text_block = format_rules_block([])
            else:
                rules = await asyncio.to_thread(
                    self._kb_store.get_relevant_rules, body
                )
                text_block = format_rules_block(rules)
                # Fire-and-forget increment. Don't block the response, don't
                # propagate DB errors — stats tracking must never break
                # Holmes' next round.
                if rules:
                    rule_ids = [r["rule_id"] for r in rules]
                    asyncio.ensure_future(
                        asyncio.to_thread(self._kb_store.record_matches, rule_ids)
                    )
        except Exception as e:
            logger.warning("kb_match_failed: %s", e)
            try:
                from src.bot.services.kb_store import format_rules_block
                text_block = format_rules_block([])
            except Exception:
                text_block = "===== KB RULES (error) =====\n===== END KB RULES ====="
        return web.Response(text=text_block, content_type="text/plain")

    async def kb_stats_handler(self, request: web.Request) -> web.Response:
        """GET /kb_stats — top-N rules by match_count desc, for #/kbstats
        admin slash command (out of scope: endpoint is the foundation).
        Loopback-only.
        """
        if request.remote not in _LOOPBACK:
            return web.Response(status=403, text="forbidden")
        if self._kb_store is None:
            return web.json_response({"rules": []})
        try:
            limit = int(request.query.get("limit", "20"))
        except ValueError:
            limit = 20
        try:
            top = await asyncio.to_thread(self._kb_store.top_by_match_count, limit)
        except Exception as e:
            logger.warning("kb_stats_failed: %s", e)
            top = []
        # Convert datetime to ISO for JSON.
        for r in top:
            if "last_matched_at" in r and r["last_matched_at"]:
                r["last_matched_at"] = r["last_matched_at"].isoformat()
        return web.json_response({"rules": top})


# Global health checker instance
health_checker = HealthChecker()