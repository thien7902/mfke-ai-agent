"""Health check HTTP server for liveness/readiness probes."""
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from aiohttp import web

from src.bot.utils.config import config

logger = logging.getLogger(__name__)


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

    def create_app(self) -> web.Application:
        """Create the aiohttp application with health endpoints."""
        app = web.Application()
        app.router.add_get("/health", self.health_handler)
        app.router.add_get("/live", self.liveness_handler)
        app.router.add_get("/ready", self.readiness_handler)
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
        # This endpoint should always return 200 if the process is running
        # Kubernetes will restart the container if this fails
        return web.json_response({
            "status": "alive",
            "bot": self.status.bot_alive,
        })

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


# Global health checker instance
health_checker = HealthChecker()