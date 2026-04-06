"""
UnslothCraft Backend entry point.
Usage: python run.py [--port 8001] [--host 0.0.0.0]
"""
import sys
import os
import argparse
import socket
import time
from pathlib import Path

# Ensure backend dir is importable
_here = str(Path(__file__).parent)
if _here not in sys.path:
    sys.path.insert(0, _here)


def _is_port_free(host: str, port: int) -> bool:
    """Return True if the port is available."""
    try:
        with socket.create_connection((host if host != "0.0.0.0" else "127.0.0.1", port), timeout=1):
            return False  # connection succeeded → port is in use
    except (ConnectionRefusedError, OSError):
        return True


def _find_free_port(start: int, max_attempts: int = 20) -> int:
    for offset in range(max_attempts):
        port = start + offset
        if _is_port_free("127.0.0.1", port):
            return port
    raise RuntimeError(f"No free port found starting from {start}")


def run_server(host: str = "0.0.0.0", port: int = 8001):
    import nest_asyncio
    nest_asyncio.apply()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)

    if not _is_port_free("127.0.0.1", port):
        print(f"Port {port} is busy — searching for a free port...")
        port = _find_free_port(port + 1)
        print(f"Using port {port}")

    import uvicorn
    from main import app

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",   # only warnings/errors — hides all HTTP 200 OK lines
        access_log=False,      # disables "GET /api/train/status 200 OK" spam
    )
    server = uvicorn.Server(config)

    print(f"\n{'='*55}")
    print(f"  UnslothCraft Backend")
    print(f"  http://localhost:{port}")
    print(f"  API docs: http://localhost:{port}/docs")
    print(f"{'='*55}\n")

    import threading
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # Keep main thread alive
    try:
        while t.is_alive():
            t.join(timeout=1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.should_exit = True
        t.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UnslothCraft Backend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
