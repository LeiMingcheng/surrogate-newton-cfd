"""Dependency-light loopback HTTP server for the interactive CFD demo."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import signal
import threading
import time
from collections import defaultdict, deque
from http.cookies import SimpleCookie
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from demo.compute import MAX_UPLOAD_BYTES, DemoEngine
from demo.jobs import (
    JobNotFoundError,
    JobScheduler,
    JobStateError,
    QueueCapacityError,
)

STATIC_ROOT = Path(__file__).resolve().parent / "static"
STATIC_ROUTES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/demo": "demo.html",
    "/demo/": "demo.html",
    "/demo.html": "demo.html",
    "/styles.css": "styles.css",
    "/project.css": "project.css",
    "/app.js": "app.js",
    "/example-airfoil.dat": "example-airfoil.txt",
    "/assets/workflow.svg": "assets/workflow.svg",
    "/assets/benchmark.svg": "assets/benchmark.svg",
    "/assets/recovery.svg": "assets/recovery.svg",
    "/assets/paper-workflow.jpg": "assets/paper-workflow.jpg",
    "/assets/paper-ood-benchmark.jpg": "assets/paper-ood-benchmark.jpg",
    "/assets/paper-recovery.jpg": "assets/paper-recovery.jpg",
    "/assets/paper-optimization.png": "assets/paper-optimization.png",
    **{
        f"/assets/figures/{name}.png": f"assets/figures/{name}.png"
        for name in (
            "fig1_workflow",
            "fig2_a_dataset_geometries",
            "fig2_b_geometry_ood_score_distribution",
            "fig3_a_recovery_comparison",
            "fig3_c_ellipse_pressure_recovery",
            "fig5_a_optimization_curves_2x2_nk",
            "fig6_a_geometry_ood",
            "fig6_c_surface_cp_recovery",
            "fig6_d_pressure_flow_recovery",
        )
    },
    **{
        f"/assets/figure-sources/{name}.pdf": f"assets/figure-sources/{name}.pdf"
        for name in (
            "fig1_workflow",
            "fig2_a_dataset_geometries",
            "fig2_b_geometry_ood_score_distribution",
            "fig3_a_recovery_comparison",
            "fig3_c_ellipse_pressure_recovery",
            "fig5_a_optimization_curves_2x2_nk",
            "fig6_a_geometry_ood",
            "fig6_c_surface_cp_recovery",
            "fig6_d_pressure_flow_recovery",
        )
    },
}
CASE_ROUTE = re.compile(r"^/api/cases/(case_[A-Za-z0-9_]+)$")
ACTION_ROUTE = re.compile(r"^/api/cases/(case_[A-Za-z0-9_]+)/(recover|reference)$")
JOB_ROUTE = re.compile(r"^/api/jobs/(job_[0-9a-f]{32})(/result)?$")
UIUC_AIRFOIL_ROUTE = re.compile(r"^/api/uiuc/airfoil/([A-Za-z0-9_.+()-]+\.dat)$", re.I)
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".pdf": "application/pdf",
    ".json": "application/json; charset=utf-8",
    ".dat": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}


class SlidingWindowRateLimiter:
    """Small in-process abuse guard; the scheduler remains the capacity authority."""

    def __init__(self) -> None:
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def require(self, client_id: str, category: str, *, limit: int, window: float) -> None:
        now = time.monotonic()
        key = (client_id, category)
        with self._lock:
            events = self._events[key]
            while events and events[0] <= now - window:
                events.popleft()
            if len(events) >= limit:
                raise QueueCapacityError("Too many requests; please retry shortly.")
            events.append(now)


class DemoRequestHandler(BaseHTTPRequestHandler):
    """Serve the browser app and expose a narrow same-origin JSON API."""

    engine: DemoEngine
    scheduler: JobScheduler
    session_secret: bytes | None = None
    enforce_session_ownership = False
    secure_session_cookie = False
    public_origin: str | None = None
    rate_limiter: SlidingWindowRateLimiter | None = None
    server_version = "Surrogate-Newton-Demo/0"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[demo] {self.address_string()} {format % args}")

    def _headers(self, status: int, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'",
        )
        pending_cookie = getattr(self, "_pending_session_cookie", None)
        if pending_cookie:
            self.send_header("Set-Cookie", pending_cookie)
        self.end_headers()

    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._headers(int(status), "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _error(self, exc: Exception) -> None:
        message_text = str(exc)
        contains_absolute_path = any(
            Path(part.strip("'\"(),:;")).is_absolute()
            for part in message_text.split()
        )
        if isinstance(exc, QueueCapacityError):
            status = HTTPStatus.TOO_MANY_REQUESTS
            message = str(exc)
        elif isinstance(exc, JobNotFoundError):
            status = HTTPStatus.NOT_FOUND
            message = str(exc)
        elif isinstance(exc, JobStateError):
            status = HTTPStatus.CONFLICT
            message = str(exc)
        elif isinstance(exc, ValueError):
            status = HTTPStatus.BAD_REQUEST
            message = "Request validation failed." if contains_absolute_path else message_text
        elif isinstance(exc, (FileNotFoundError, KeyError)):
            status = HTTPStatus.NOT_FOUND
            message = (
                "The requested demo resource was not found."
                if contains_absolute_path
                else message_text.strip("'")
            )
        elif isinstance(exc, (ConnectionError, TimeoutError, RuntimeError)):
            status = HTTPStatus.SERVICE_UNAVAILABLE
            message = "The requested compute runtime is unavailable."
        else:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            message = "The demo could not complete the request."
        self._json(
            {"ok": False, "error": message},
            status=status,
        )

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES + 100_000:
            raise ValueError("Request body is empty or exceeds the local-demo limit.")
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise ValueError("The demo API accepts application/json requests.")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON request must be an object.")
        return payload

    def _client_id(self) -> str:
        session_id = self._session_id()
        if session_id is not None:
            return session_id
        remote = str(self.client_address[0])
        if ipaddress.ip_address(remote).is_loopback:
            forwarded = self.headers.get("X-Forwarded-For", "").split(",", 1)[0].strip()
            if forwarded:
                try:
                    return str(ipaddress.ip_address(forwarded))
                except ValueError:
                    pass
        return remote

    def _session_id(self) -> str | None:
        if self.session_secret is None:
            return None
        cached = getattr(self, "_verified_session_id", None)
        if cached:
            return str(cached)
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except Exception:
            cookie = SimpleCookie()
        value = cookie.get("sn_session")
        token = "" if value is None else value.value.rsplit(".", 1)[0]
        supplied_signature = "" if value is None else value.value.rsplit(".", 1)[-1]
        expected_signature = hmac.new(
            self.session_secret, token.encode("ascii", errors="ignore"), hashlib.sha256
        ).hexdigest()
        if (
            not re.fullmatch(r"[A-Za-z0-9_-]{43}", token)
            or not hmac.compare_digest(supplied_signature, expected_signature)
        ):
            token = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
            signature = hmac.new(
                self.session_secret, token.encode("ascii"), hashlib.sha256
            ).hexdigest()
            secure = "; Secure" if self.secure_session_cookie else ""
            self._pending_session_cookie = (
                f"sn_session={token}.{signature}; Path=/; HttpOnly; SameSite=Strict; "
                f"Max-Age=604800{secure}"
            )
        self._verified_session_id = token
        return token

    def _require_same_origin(self) -> None:
        if self.public_origin is None:
            return
        origin = self.headers.get("Origin")
        if origin is not None and origin.rstrip("/") != self.public_origin.rstrip("/"):
            raise ValueError("Cross-origin requests are not accepted.")

    def _require_rate_limit(self, category: str, *, limit: int) -> None:
        if self.rate_limiter is not None:
            self.rate_limiter.require(
                self._client_id(), category, limit=limit, window=60.0
            )

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        try:
            if self.session_secret is not None:
                self._session_id()
            if path == "/api/health/live":
                self._json({"ok": True, "status": "live"})
                return
            if path == "/api/health/ready":
                status = self.engine.status()
                ready = bool(status["surrogate_online"] and status["solver_ready"])
                self._json(
                    {"ok": ready, "status": "ready" if ready else "not_ready"},
                    status=HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            if path == "/api/status":
                scheduler_status = self.scheduler.stats()
                if self.enforce_session_ownership:
                    for running in scheduler_status["running_jobs"]:
                        running.pop("job_id", None)
                    scheduler_status["running"] = (
                        scheduler_status["running_jobs"][0]
                        if scheduler_status["running_jobs"]
                        else None
                    )
                self._json(
                    {
                        "ok": True,
                        **self.engine.status(),
                        "scheduler": scheduler_status,
                    }
                )
                return
            if path == "/api/presets":
                self._json({"ok": True, **self.engine.presets()})
                return
            if path == "/api/uiuc/catalog":
                self._json({"ok": True, **self.engine.uiuc_catalog()})
                return
            uiuc_match = UIUC_AIRFOIL_ROUTE.match(path)
            if uiuc_match:
                self._json({"ok": True, **self.engine.uiuc_airfoil(uiuc_match.group(1))})
                return
            job_match = JOB_ROUTE.match(path)
            if job_match:
                self._require_rate_limit("job-read", limit=300)
                job_id, result_suffix = job_match.groups()
                client_id = self._client_id() if self.enforce_session_ownership else None
                if result_suffix:
                    self._json(
                        {"ok": True, "result": self.scheduler.result(job_id, client_id=client_id)}
                    )
                else:
                    self._json(
                        {"ok": True, "job": self.scheduler.get(job_id, client_id=client_id)}
                    )
                return
            match = CASE_ROUTE.match(path)
            if match:
                if self.enforce_session_ownership:
                    self.scheduler.authorize_case(match.group(1), self._client_id())
                self._json({"ok": True, **self.engine.get_case(match.group(1))})
                return
            self._serve_static(path)
        except Exception as exc:
            self._error(exc)

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        try:
            self._require_same_origin()
            if path == "/api/geometry/project":
                self._require_rate_limit("geometry-project", limit=180)
            else:
                self._require_rate_limit("post", limit=30)
            action_match = ACTION_ROUTE.match(path)
            if path not in {
                "/api/geometry/import",
                "/api/geometry/project",
                "/api/jobs",
                "/api/predict",
                "/api/mesh",
            } and action_match is None:
                self._json(
                    {"ok": False, "error": "Unknown API route."},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            payload = self._body()
            if path == "/api/geometry/import":
                result = self.engine.import_coordinates(
                    str(payload["content"]),
                    str(payload.get("filename") or "uploaded-airfoil.dat"),
                )
            elif path == "/api/geometry/project":
                result = self.engine.project_geometry(
                    payload["x"],
                    payload["upper"],
                    payload["lower"],
                    str(payload.get("name") or "Edited airfoil"),
                )
            elif path == "/api/jobs":
                job = self.scheduler.submit(
                    str(payload.get("action") or ""),
                    payload.get("payload") or {},
                    client_id=self._client_id(),
                )
                self._json({"ok": True, "job": job}, status=HTTPStatus.ACCEPTED)
                return
            elif path in {"/api/predict", "/api/mesh"}:
                job = self.scheduler.submit(
                    "predict" if path == "/api/predict" else "mesh",
                    payload,
                    client_id=self._client_id(),
                )
                self._json({"ok": True, "job": job}, status=HTTPStatus.ACCEPTED)
                return
            else:
                case_id, action = action_match.groups()
                job = self.scheduler.submit(
                    action,
                    {"case_id": case_id, **payload},
                    client_id=self._client_id(),
                )
                self._json({"ok": True, "job": job}, status=HTTPStatus.ACCEPTED)
                return
            self._json({"ok": True, **result})
        except Exception as exc:
            self._error(exc)

    def do_DELETE(self) -> None:
        path = unquote(urlparse(self.path).path)
        try:
            self._require_same_origin()
            self._require_rate_limit("delete", limit=30)
            job_match = JOB_ROUTE.match(path)
            if job_match is None or job_match.group(2):
                self._json(
                    {"ok": False, "error": "Unknown API route."},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            client_id = self._client_id() if self.enforce_session_ownership else None
            self._json(
                {
                    "ok": True,
                    "job": self.scheduler.cancel(job_match.group(1), client_id=client_id),
                }
            )
        except Exception as exc:
            self._error(exc)

    def _serve_static(self, request_path: str) -> None:
        if request_path in STATIC_ROUTES:
            path = STATIC_ROOT / STATIC_ROUTES[request_path]
        else:
            self._json(
                {"ok": False, "error": "Not found."},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        body = path.read_bytes()
        self._headers(
            HTTPStatus.OK,
            CONTENT_TYPES.get(path.suffix, "application/octet-stream"),
            len(body),
        )
        self.wfile.write(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Surrogate-Newton CFD demo.")
    parser.add_argument("--host", default=os.environ.get("DEMO_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DEMO_WEB_PORT", "8080")))
    parser.add_argument(
        "--surrogate-host",
        default=os.environ.get("DEMO_SURROGATE_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--surrogate-port",
        type=int,
        default=int(os.environ.get("DEMO_SURROGATE_PORT", "65432")),
    )
    parser.add_argument("--runtime-root", default=os.environ.get("DEMO_RUNTIME_ROOT"))
    parser.add_argument(
        "--airfoil-library-root",
        default=os.environ.get("DEMO_AIRFOIL_LIBRARY_ROOT"),
    )
    parser.add_argument("--ood-asset-root", default=os.environ.get("DEMO_OOD_ASSET_ROOT"))
    parser.add_argument(
        "--mpi-launcher", default=os.environ.get("DEMO_MPI_LAUNCHER", "auto")
    )
    parser.add_argument(
        "--mpi-ranks", type=int, default=int(os.environ.get("DEMO_MPI_RANKS", "8"))
    )
    parser.add_argument(
        "--resident-idle-timeout-sec",
        type=float,
        default=float(os.environ.get("DEMO_RESIDENT_IDLE_TIMEOUT_SEC", "0")),
        help="Resident MPI worker idle timeout in seconds; zero keeps workers alive.",
    )
    parser.add_argument(
        "--ank-nk-max-work",
        type=int,
        default=int(os.environ.get("DEMO_ANK_NK_MAX_WORK", "1000")),
    )
    parser.add_argument(
        "--ank-nk-time-limit-s",
        type=float,
        default=float(os.environ.get("DEMO_ANK_NK_TIME_LIMIT_S", "10")),
    )
    parser.add_argument(
        "--ank-nk-switch-tolerance",
        type=float,
        default=float(os.environ.get("DEMO_ANK_NK_SWITCH_TOL", "1e-4")),
    )
    parser.add_argument("--model-dir", default=os.environ.get("SURROGATE_NEWTON_MODEL_DIR"))
    parser.add_argument("--model-config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--statistics")
    parser.add_argument("--device", default=os.environ.get("DEMO_DEVICE", "cuda:0"))
    parser.add_argument(
        "--max-pending-jobs",
        type=int,
        default=int(os.environ.get("DEMO_MAX_PENDING_JOBS", "64")),
    )
    parser.add_argument(
        "--heavy-job-concurrency",
        type=int,
        choices=(1, 2),
        default=int(os.environ.get("DEMO_HEAVY_JOB_CONCURRENCY", "1")),
    )
    parser.add_argument(
        "--nk-burst-limit",
        type=int,
        default=int(os.environ.get("DEMO_NK_BURST_LIMIT", "3")),
    )
    parser.add_argument(
        "--cold-start-max-wait-sec",
        type=float,
        default=float(os.environ.get("DEMO_COLD_START_MAX_WAIT_SEC", "300")),
    )
    parser.add_argument(
        "--max-pending-jobs-per-client",
        type=int,
        default=int(os.environ.get("DEMO_MAX_PENDING_JOBS_PER_CLIENT", "4")),
    )
    parser.add_argument(
        "--job-result-ttl-sec",
        type=float,
        default=float(os.environ.get("DEMO_JOB_RESULT_TTL_SEC", "86400")),
    )
    parser.add_argument(
        "--job-cleanup-interval-sec",
        type=float,
        default=float(os.environ.get("DEMO_JOB_CLEANUP_INTERVAL_SEC", "60")),
    )
    parser.add_argument(
        "--job-max-result-bytes",
        type=int,
        default=int(os.environ.get("DEMO_JOB_MAX_RESULT_BYTES", str(64 * 1024 * 1024))),
    )
    parser.add_argument(
        "--mesh-job-timeout-sec",
        type=float,
        default=float(os.environ.get("DEMO_MESH_JOB_TIMEOUT_SEC", "600")),
    )
    parser.add_argument(
        "--predict-job-timeout-sec",
        type=float,
        default=float(os.environ.get("DEMO_PREDICT_JOB_TIMEOUT_SEC", "600")),
    )
    parser.add_argument(
        "--recover-job-timeout-sec",
        type=float,
        default=float(os.environ.get("DEMO_RECOVER_JOB_TIMEOUT_SEC", "7200")),
    )
    parser.add_argument(
        "--reference-job-timeout-sec",
        type=float,
        default=float(os.environ.get("DEMO_REFERENCE_JOB_TIMEOUT_SEC", "7200")),
    )
    parser.add_argument(
        "--case-ttl-sec",
        type=float,
        default=float(os.environ.get("DEMO_CASE_TTL_SEC", "86400")),
    )
    parser.add_argument(
        "--cancel-grace-sec",
        type=float,
        default=float(os.environ.get("DEMO_CANCEL_GRACE_SEC", "30")),
    )
    parser.add_argument(
        "--session-secret-file",
        default=os.environ.get("DEMO_SESSION_SECRET_FILE"),
    )
    parser.add_argument(
        "--public-origin",
        default=os.environ.get("DEMO_PUBLIC_ORIGIN"),
    )
    parser.add_argument(
        "--public-mode",
        action="store_true",
        default=os.environ.get("DEMO_PUBLIC_MODE", "0") == "1",
    )
    parser.add_argument(
        "--allow-insecure-public-http",
        action="store_true",
        default=os.environ.get("DEMO_ALLOW_INSECURE_PUBLIC_HTTP", "0") == "1",
        help=(
            "Allow public-mode session cookies over HTTP when the hosting provider "
            "cannot expose HTTPS. This is an explicit preview-only downgrade."
        ),
    )
    parser.add_argument(
        "--hard-timeout-exit",
        action="store_true",
        default=os.environ.get("DEMO_HARD_TIMEOUT_EXIT", "0") == "1",
        help="Exit the process when a native job exceeds its deadline so the supervisor restarts it.",
    )
    parser.add_argument("--skip-prewarm", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    if args.host not in loopback_hosts:
        raise ValueError(
            "The initial demo is intentionally local-only; bind to 127.0.0.1 or localhost."
        )
    if args.surrogate_host not in loopback_hosts:
        raise ValueError("The native Surrogate service must remain on the loopback interface.")
    session_secret: bytes | None = None
    if args.session_secret_file:
        secret_path = Path(args.session_secret_file).expanduser()
        session_secret = secret_path.read_bytes().strip()
        if len(session_secret) < 32:
            raise ValueError("The session secret file must contain at least 32 random bytes.")
    if args.public_mode and (session_secret is None or not args.public_origin):
        raise ValueError("Public mode requires a session secret file and DEMO_PUBLIC_ORIGIN.")
    if args.allow_insecure_public_http and not args.public_mode:
        raise ValueError("Insecure public HTTP is valid only together with public mode.")
    if args.public_mode:
        public_scheme = urlparse(args.public_origin).scheme.lower()
        if args.allow_insecure_public_http:
            if public_scheme != "http":
                raise ValueError("Insecure public HTTP requires an http:// public origin.")
        elif public_scheme != "https":
            raise ValueError(
                "Public mode requires an https:// origin unless the explicit "
                "preview-only HTTP downgrade is enabled."
            )
    engine_options: dict[str, Any] = {
        "runtime_root": args.runtime_root,
        "surrogate_host": args.surrogate_host,
        "surrogate_port": args.surrogate_port,
        "airfoil_library_root": args.airfoil_library_root,
        "ood_asset_root": args.ood_asset_root,
        "mpi_launcher": args.mpi_launcher,
        "mpi_ranks": args.mpi_ranks,
        "resident_idle_timeout_sec": args.resident_idle_timeout_sec,
        "ank_nk_max_work": args.ank_nk_max_work,
        "ank_nk_time_limit_s": args.ank_nk_time_limit_s,
        "ank_nk_switch_tolerance": args.ank_nk_switch_tolerance,
        "model_dir": args.model_dir,
        "checkpoint": args.checkpoint,
        "statistics": args.statistics,
        "device": args.device,
    }
    if args.model_config:
        engine_options["model_config"] = args.model_config
    if args.runtime_root:
        engine_options["resident_pool_root"] = (
            Path(args.runtime_root).expanduser() / "resident_adflow" / "worker_0000"
        )
    engines = [DemoEngine(**engine_options)]
    for worker_id in range(1, args.heavy_job_concurrency):
        worker_options = {
            **engine_options,
            "runtime_root": engines[0].runtime_root,
            "resident_pool_root": (
                engines[0].runtime_root
                / "resident_adflow"
                / f"worker_{worker_id:04d}"
            ),
        }
        engines.append(DemoEngine(**worker_options))
    engine = engines[0]
    scheduler: JobScheduler | None = None
    try:
        if not args.skip_prewarm:
            print("Prewarming resident MPI workers, pyHyp, and Surrogate…", flush=True)
            for worker_id, worker_engine in enumerate(engines):
                timing = worker_engine.prewarm()
                print(
                    f"Prewarm worker {worker_id}: "
                    f"MPI workers={timing['resident_pool_wall_sec']:.2f}s, "
                    f"pyHyp={timing['pyhyp_wall_sec']:.2f}s, "
                    f"model={timing['inference_wall_sec']:.2f}s",
                    flush=True,
                )
        def hard_timeout_exit(event: dict[str, Any]) -> None:
            print(
                json.dumps({"event": "hard_job_watchdog", **event}, sort_keys=True),
                flush=True,
            )
            os._exit(70)

        scheduler = JobScheduler(
            engines,
            runtime_root=engine.runtime_root / "scheduler",
            max_pending_jobs=args.max_pending_jobs,
            max_pending_jobs_per_client=args.max_pending_jobs_per_client,
            result_ttl_sec=args.job_result_ttl_sec,
            cleanup_interval_sec=args.job_cleanup_interval_sec,
            max_result_bytes=args.job_max_result_bytes,
            nk_burst_limit=args.nk_burst_limit,
            cold_start_max_wait_sec=args.cold_start_max_wait_sec,
            case_root=engine.case_root,
            mesh_root=engine.mesh_root,
            solver_prepare_root=engine.runtime_root / "solver_prepare",
            case_ttl_sec=args.case_ttl_sec,
            enforce_case_ownership=args.public_mode,
            hard_timeout_handler=hard_timeout_exit if args.hard_timeout_exit else None,
            cancel_grace_sec=args.cancel_grace_sec,
            action_timeouts={
                "mesh": args.mesh_job_timeout_sec,
                "predict": args.predict_job_timeout_sec,
                "recover": args.recover_job_timeout_sec,
                "reference": args.reference_job_timeout_sec,
            },
        )
        handler = type(
            "ConfiguredDemoRequestHandler",
            (DemoRequestHandler,),
            {
                "engine": engine,
                "scheduler": scheduler,
                "session_secret": session_secret,
                "enforce_session_ownership": args.public_mode,
                "secure_session_cookie": (
                    args.public_mode and not args.allow_insecure_public_http
                ),
                "public_origin": args.public_origin if args.public_mode else None,
                "rate_limiter": SlidingWindowRateLimiter() if args.public_mode else None,
            },
        )
        server = ThreadingHTTPServer((args.host, int(args.port)), handler)
        server.daemon_threads = True

        def stop_server(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop_server)
        print(
            json.dumps(
                {
                    "event": "demo_server_starting",
                    "web_host": args.host,
                    "web_port": args.port,
                    "surrogate_host": args.surrogate_host,
                    "surrogate_port": args.surrogate_port,
                    "mpi_ranks": args.mpi_ranks,
                    "heavy_job_concurrency": args.heavy_job_concurrency,
                    "nk_burst_limit": args.nk_burst_limit,
                    "cold_start_max_wait_sec": args.cold_start_max_wait_sec,
                    "max_pending_jobs": args.max_pending_jobs,
                    "public_mode": args.public_mode,
                    "insecure_public_http": args.allow_insecure_public_http,
                    "hard_timeout_exit": args.hard_timeout_exit,
                    "prewarm": not args.skip_prewarm,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        print(f"Project page: http://{args.host}:{args.port}/", flush=True)
        print(f"Interactive demo: http://{args.host}:{args.port}/demo", flush=True)
        print("Press Ctrl-C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        scheduler_stopped = scheduler is None or scheduler.close()
        if scheduler_stopped:
            for worker_engine in engines:
                worker_engine.close()
        else:
            print(
                "A compute job is still stopping; the process will release its runtime resources.",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
