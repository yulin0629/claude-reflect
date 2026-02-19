#!/usr/bin/env python3
"""Client for the embedding classification daemon.

Communicates with embedding_server.py over Unix domain socket.
Handles server startup, connection, and graceful degradation.
"""
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

SOCKET_PATH = "/tmp/claude-reflect-embedding.sock"
PID_FILE = Path.home() / ".claude" / "embedding-server.pid"

# Timeouts
CONNECT_TIMEOUT = 2.0
READ_TIMEOUT = 5.0
SERVER_START_TIMEOUT = 10.0  # Max wait for server to become ready
SERVER_START_POLL = 0.3  # Poll interval when waiting for server

# Path to the server script (relative to this file)
_SERVER_SCRIPT = Path(__file__).resolve().parent.parent / "embedding_server.py"


def _is_server_running() -> bool:
    """Check if the embedding server process is alive."""
    if not PID_FILE.exists():
        return False

    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, IOError):
        return False

    # Check if process exists (signal 0 doesn't kill, just checks)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        # Process doesn't exist, clean up stale PID file
        try:
            PID_FILE.unlink()
        except OSError:
            pass
        return False


def _is_socket_ready() -> bool:
    """Check if the Unix socket exists and is connectable."""
    if not Path(SOCKET_PATH).exists():
        return False

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        sock.connect(SOCKET_PATH)
        sock.close()
        return True
    except (OSError, socket.error):
        return False


def _find_python() -> str:
    """Find a suitable Python interpreter that has onnxruntime installed."""
    # Check if uv is available (preferred for dependency management)
    for uv_path in ["uv", "/opt/homebrew/bin/uv", os.path.expanduser("~/.local/bin/uv")]:
        try:
            result = subprocess.run(
                [uv_path, "--version"],
                capture_output=True, timeout=2,
            )
            if result.returncode == 0:
                return uv_path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return ""


def ensure_server_running() -> bool:
    """Ensure the embedding server is running. Start it if not.

    Returns True if server is ready, False if unable to start.
    """
    # Fast path: already running and socket ready
    if _is_server_running() and _is_socket_ready():
        return True

    # Clean up stale socket
    if Path(SOCKET_PATH).exists() and not _is_server_running():
        try:
            Path(SOCKET_PATH).unlink()
        except OSError:
            pass

    # Find Python/uv and start server
    uv_path = _find_python()
    if not uv_path:
        return False

    if not _SERVER_SCRIPT.exists():
        return False

    try:
        # Start server in background via uv
        cmd = [
            uv_path, "run",
            "--with", "onnxruntime",
            "--with", "tokenizers",
            "--with", "numpy",
            "python", str(_SERVER_SCRIPT),
        ]

        # Detach from parent process
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False

    # Wait for server to become ready
    deadline = time.monotonic() + SERVER_START_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(SERVER_START_POLL)
        if _is_socket_ready():
            return True

    return False


def _send_request(sock: socket.socket, data: dict) -> None:
    """Send a length-prefixed JSON request."""
    payload = json.dumps(data).encode("utf-8")
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _recv_response(sock: socket.socket) -> dict:
    """Receive a length-prefixed JSON response."""
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("Server disconnected")
        header += chunk

    msg_len = struct.unpack(">I", header)[0]
    if msg_len > 65536:
        raise ValueError(f"Response too large: {msg_len}")

    data = b""
    while len(data) < msg_len:
        chunk = sock.recv(min(msg_len - len(data), 4096))
        if not chunk:
            raise ConnectionError("Server disconnected")
        data += chunk

    return json.loads(data.decode("utf-8"))


def classify_via_daemon(text: str) -> Tuple[Optional[str], str, float, str, int]:
    """Query the embedding server for classification.

    Returns detect_patterns-compatible 5-tuple:
    (type, patterns, confidence, sentiment, decay_days)

    On any failure, returns (None, "", 0.0, "correction", 90) for silent degradation.
    """
    try:
        if not ensure_server_running():
            return (None, "", 0.0, "correction", 90)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(CONNECT_TIMEOUT)
        sock.connect(SOCKET_PATH)

        sock.settimeout(READ_TIMEOUT)
        _send_request(sock, {"text": text})
        result = _recv_response(sock)
        sock.close()

        if "error" in result:
            return (None, "", 0.0, "correction", 90)

        return (
            result.get("type"),
            result.get("patterns", ""),
            result.get("confidence", 0.0),
            result.get("sentiment", "correction"),
            result.get("decay_days", 90),
        )

    except Exception:
        return (None, "", 0.0, "correction", 90)


def stop_server() -> bool:
    """Stop the embedding server if running. Returns True if stopped."""
    if not PID_FILE.exists():
        return False

    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        os.kill(pid, signal.SIGTERM)
        # Wait a moment for cleanup
        time.sleep(0.5)
        return True
    except (ValueError, IOError, OSError):
        return False
