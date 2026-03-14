from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sas94_search_api.app import add_retrieval_args, build_retrieval_config
from sas94_search_api.retrieval import env_default, load_dotenv
from sas94_search_api.search_service import run_search


def parse_search_api_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the SAS search API.")
    parser.add_argument("--host", default=env_default("RAG_SEARCH_API_HOST", "127.0.0.1"), help="Bind host.")
    parser.add_argument(
        "--port",
        type=int,
        default=int(env_default("RAG_SEARCH_API_PORT", "8788")),
        help="Bind port.",
    )
    add_retrieval_args(parser, top_k_default=5, top_k_help="Default top-k for API requests.")
    return parser.parse_args()


def build_search_config_from_request(server_args: argparse.Namespace, payload: dict[str, object]):
    args = argparse.Namespace(**vars(server_args))
    args.mode = str(payload.get("mode", server_args.mode))
    args.top_k = int(payload.get("top_k", server_args.top_k))
    args.rerank = bool(payload.get("rerank", server_args.rerank))
    args.no_term_expansion = bool(payload.get("no_term_expansion", server_args.no_term_expansion))
    args.rerank_limit = int(payload.get("rerank_limit", server_args.rerank_limit))
    config = build_retrieval_config(args)
    config.dense_limit = max(args.top_k * 4, server_args.dense_limit)
    config.lexical_limit = max(args.top_k * 4, server_args.lexical_limit)
    return config


def make_search_handler(server_args: argparse.Namespace):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json({"ok": True})
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/api/search":
                self._send_json({"error": "not found"}, status=404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                self._send_json({"error": "invalid json"}, status=400)
                return

            query = str(payload.get("query", "")).strip()
            if not query:
                self._send_json({"error": "query is required"}, status=400)
                return

            try:
                config = build_search_config_from_request(server_args, payload)
                search_response = run_search(query, config)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
                return

            self._send_json({"query": search_response.query, "retrieval": search_response.retrieval})

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def serve_search_api() -> int:
    load_dotenv()
    args = parse_search_api_args()
    server = ThreadingHTTPServer((args.host, args.port), make_search_handler(args))
    print(f"SAS search API listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
