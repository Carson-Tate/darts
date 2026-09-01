#!/usr/bin/env python3
"""Entry point. `python run.py` and open http://darts.local:8000 on your phone."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from darts.config import load_config
from darts.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Darts scoring server")
    parser.add_argument("-c", "--config", help="path to config.yaml")
    parser.add_argument("--host", help="override the bind address")
    parser.add_argument("--port", type=int, help="override the port")
    parser.add_argument("--no-vision", action="store_true", help="manual entry only")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # OpenCV's own chatter is not useful here.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    cfg = load_config(args.config)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port

    app = create_app(args.config)
    if args.no_vision:
        # Set on the app's own config -- create_app loads its own copy, and the
        # cameras aren't opened until the lifespan hook runs.
        app.state.hub.cfg.vision.enabled = False

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
