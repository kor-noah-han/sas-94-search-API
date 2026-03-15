from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sas94_search_api.app import add_retrieval_args, build_retrieval_config
from sas94_search_api.logging_utils import configure_logging, get_logger
from sas94_search_api.retrieval import env_default, load_dotenv, load_section_routes
from sas94_search_api.search_service import config_cache_dict, run_search


HTTP_CACHE_SIZE = int(env_default("RAG_SEARCH_HTTP_CACHE_SIZE", "128") or "128")
HTTP_CACHE_TTL_SECONDS = int(env_default("RAG_SEARCH_HTTP_CACHE_TTL", "30") or "30")
_HTTP_RESPONSE_CACHE: OrderedDict[str, tuple[float, dict[str, object]]] = OrderedDict()
VALID_MODES = {"dense", "lexical", "hybrid"}
TRUTHY_VALUES = {"1", "true", "yes", "on"}
FALSY_VALUES = {"0", "false", "no", "off"}
LOGGER = get_logger(__name__)


class RequestValidationError(ValueError):
    pass


def require_object(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise RequestValidationError("json body must be an object")
    return payload


def parse_choice(payload: dict[str, object], key: str, default: str, choices: set[str]) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str):
        raise RequestValidationError(f"{key} must be one of: {', '.join(sorted(choices))}")
    normalized = value.strip().lower()
    if normalized not in choices:
        raise RequestValidationError(f"{key} must be one of: {', '.join(sorted(choices))}")
    return normalized


def parse_int(payload: dict[str, object], key: str, default: int, *, minimum: int = 1) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool):
        raise RequestValidationError(f"{key} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(f"{key} must be an integer") from exc
    if parsed < minimum:
        raise RequestValidationError(f"{key} must be >= {minimum}")
    return parsed


def parse_bool(payload: dict[str, object], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUTHY_VALUES:
            return True
        if normalized in FALSY_VALUES:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise RequestValidationError(f"{key} must be a boolean")


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
    args.mode = parse_choice(payload, "mode", server_args.mode, VALID_MODES)
    args.top_k = parse_int(payload, "top_k", server_args.top_k)
    args.rerank = parse_bool(payload, "rerank", server_args.rerank)
    args.no_term_expansion = parse_bool(payload, "no_term_expansion", server_args.no_term_expansion)
    args.rerank_limit = parse_int(payload, "rerank_limit", server_args.rerank_limit)
    config = build_retrieval_config(args)
    config.dense_limit = max(args.top_k * 4, server_args.dense_limit)
    config.lexical_limit = max(args.top_k * 4, server_args.lexical_limit)
    return config


def cache_key_for_request(query: str, config) -> str:
    return json.dumps(
        {
            "query": query,
            "config": config_cache_dict(config),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def get_cached_response(cache_key: str) -> dict[str, object] | None:
    cached = _HTTP_RESPONSE_CACHE.get(cache_key)
    if cached is None:
        return None
    expires_at, payload = cached
    if expires_at < time.time():
        _HTTP_RESPONSE_CACHE.pop(cache_key, None)
        return None
    _HTTP_RESPONSE_CACHE.move_to_end(cache_key)
    return payload


def set_cached_response(cache_key: str, payload: dict[str, object]) -> None:
    _HTTP_RESPONSE_CACHE[cache_key] = (time.time() + HTTP_CACHE_TTL_SECONDS, payload)
    _HTTP_RESPONSE_CACHE.move_to_end(cache_key)
    while len(_HTTP_RESPONSE_CACHE) > HTTP_CACHE_SIZE:
        _HTTP_RESPONSE_CACHE.popitem(last=False)


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
                payload = require_object(json.loads(raw.decode("utf-8")))
            except RequestValidationError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            except Exception:
                self._send_json({"error": "invalid json"}, status=400)
                return

            query = str(payload.get("query", "")).strip()
            if not query:
                self._send_json({"error": "query is required"}, status=400)
                return

            try:
                config = build_search_config_from_request(server_args, payload)
                cache_key = cache_key_for_request(query, config)
                cached_payload = get_cached_response(cache_key)
                if cached_payload is not None:
                    self._send_json(cached_payload)
                    return
                search_response = run_search(query, config)
            except RequestValidationError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
                return

            response_payload = {"query": search_response.query, "retrieval": search_response.retrieval}
            set_cached_response(cache_key, response_payload)
            self._send_json(response_payload)

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def serve_search_api() -> int:
    load_dotenv()
    configure_logging()
    args = parse_search_api_args()
    preload_config = build_retrieval_config(args)
    load_section_routes(
        preload_config.route_index_path,
        preload_config.fts_db_path,
        preload_config.corpus_path,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_search_handler(args))
    LOGGER.info("search_api_started host=%s port=%s default_mode=%s", args.host, args.port, args.mode)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
