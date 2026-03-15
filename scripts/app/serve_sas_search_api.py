#!/usr/bin/env python3
from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

from sas94_search_api.search_api import serve_search_api


def main() -> int:
    return serve_search_api()


if __name__ == "__main__":
    raise SystemExit(main())
