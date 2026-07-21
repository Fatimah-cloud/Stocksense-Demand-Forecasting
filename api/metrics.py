"""Operational (Prometheus) metrics -- request counts and latency.

Distinct from forecast-quality metrics (MAPE/RMSE/bias), which live in
monitoring.py and are exposed via /monitoring/kpis.
"""

import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "path"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        path = request.url.path
        REQUEST_LATENCY.labels(request.method, path).observe(duration)
        REQUEST_COUNT.labels(request.method, path, response.status_code).inc()
        return response


def render_metrics() -> bytes:
    return generate_latest()


METRICS_CONTENT_TYPE = CONTENT_TYPE_LATEST
