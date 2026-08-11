"""Dependency-light loopback HTTP server for the interactive CFD demo."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from demo.compute import MAX_UPLOAD_BYTES, DemoEngine

STATIC_ROOT = Path(__file__).resolve().parent / "static"
STATIC_ROUTES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/demo": "demo.html",
    "/demo/": "demo.html",
    "/demo.html": "demo.html",
    "/styles.css": "styles.css",
    "/app.js": "app.js",
    "/example-airfoil.dat": "example-airfoil.txt",
    "/assets/workflow.svg": "assets/workflow.svg",
    "/assets/benchmark.svg": "assets/benchmark.svg",
    "/assets/recovery.svg": "assets/recovery.svg",
}
CASE_ROUTE = re.compile(r"^/api/cases/(case_[A-Za-z0-9_]+)$")
ACTION_ROUTE = re.compile(r"^/api/cases/(case_[A-Za-z0-9_]+)/(recover|reference)$")
UIUC_AIRFOIL_ROUTE = re.compile(r"^/api/uiuc/airfoil/([A-Za-z0-9_.+()-]+\.dat)$", re.I)
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".json": "application/json; charset=utf-8",
    ".dat": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}


class DemoRequestHandler(BaseHTTPRequestHandler):
    """Serve the browser app and expose a narrow same-origin JSON API."""

    engine: DemoEngine
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
        if isinstance(exc, ValueError):
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

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/api/status":
                self._json({"ok": True, **self.engine.status()})
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
            match = CASE_ROUTE.match(path)
            if match:
                self._json({"ok": True, **self.engine.get_case(match.group(1))})
                return
            self._serve_static(path)
        except Exception as exc:
            self._error(exc)

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        try:
            action_match = ACTION_ROUTE.match(path)
            if path not in {
                "/api/geometry/import",
                "/api/geometry/project",
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
            elif path == "/api/predict":
                result = self.engine.predict(
                    geometry27=payload["geometry27"],
                    mach=float(payload["mach"]),
                    aoa=float(payload["aoa"]),
                    n_inference_steps=int(payload.get("n_inference_steps", 5)),
                    name=str(payload.get("name") or "Demo airfoil"),
                )
            elif path == "/api/mesh":
                result = self.engine.prepare_mesh(
                    geometry27=payload["geometry27"],
                    name=str(payload.get("name") or "Demo airfoil"),
                )
            else:
                case_id, action = action_match.groups()
                if action == "recover":
                    result = self.engine.recover(
                        case_id,
                        cycles=int(payload.get("cycles", 6)),
                        residual_exponent=int(payload.get("residual_exponent", 8)),
                    )
                else:
                    result = self.engine.reference(
                        case_id,
                        max_cycles=int(payload.get("max_cycles", 3000)),
                    )
            self._json({"ok": True, **result})
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
    parser.add_argument("--model-dir", default=os.environ.get("SURROGATE_NEWTON_MODEL_DIR"))
    parser.add_argument("--model-config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--statistics")
    parser.add_argument("--device", default=os.environ.get("DEMO_DEVICE", "cuda:0"))
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
    engine_options: dict[str, Any] = {
        "runtime_root": args.runtime_root,
        "surrogate_host": args.surrogate_host,
        "surrogate_port": args.surrogate_port,
        "airfoil_library_root": args.airfoil_library_root,
        "ood_asset_root": args.ood_asset_root,
        "mpi_launcher": args.mpi_launcher,
        "mpi_ranks": args.mpi_ranks,
        "model_dir": args.model_dir,
        "checkpoint": args.checkpoint,
        "statistics": args.statistics,
        "device": args.device,
    }
    if args.model_config:
        engine_options["model_config"] = args.model_config
    engine = DemoEngine(
        **engine_options,
    )
    try:
        if not args.skip_prewarm:
            print("Prewarming resident MPI workers, pyHyp, and Surrogate…", flush=True)
            timing = engine.prewarm()
            print(
                "Prewarm complete: "
                f"MPI workers={timing['resident_pool_wall_sec']:.2f}s, "
                f"pyHyp={timing['pyhyp_wall_sec']:.2f}s, "
                f"model={timing['inference_wall_sec']:.2f}s",
                flush=True,
            )
        handler = type(
            "ConfiguredDemoRequestHandler",
            (DemoRequestHandler,),
            {"engine": engine},
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
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
