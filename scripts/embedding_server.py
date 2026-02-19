#!/usr/bin/env python3
"""Persistent embedding classification daemon via Unix domain socket.

Loads the ONNX model once at startup, then serves classification requests
over a Unix socket. This avoids the ~1-2s model load time on each hook call.

Protocol:
  Client sends JSON: {"text": "user message"}
  Server responds JSON: [type, patterns, confidence, sentiment, decay_days]

Lifecycle:
  - Started by ensure_embedding_server.py (SessionStart hook) or daemon_client.py
  - Auto-exits after IDLE_TIMEOUT seconds of inactivity
  - PID written to ~/.claude/embedding-server.pid
"""
import json
import os
import signal
import socket
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

from embedding_classifier import EmbeddingModel, AnchorStore, classify_message

SOCKET_PATH = "/tmp/claude-reflect-embedding.sock"
PID_FILE = Path.home() / ".claude" / "embedding-server.pid"
IDLE_TIMEOUT = 3600  # 60 minutes
MAX_MSG_SIZE = 65536


def cleanup(sock: socket.socket) -> None:
    """Clean up socket and PID file."""
    try:
        sock.close()
    except OSError:
        pass
    try:
        if Path(SOCKET_PATH).exists():
            Path(SOCKET_PATH).unlink()
    except OSError:
        pass
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except OSError:
        pass


def write_pid() -> None:
    """Write current PID to file."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")


def send_response(conn: socket.socket, data: dict) -> None:
    """Send a length-prefixed JSON response."""
    payload = json.dumps(data).encode("utf-8")
    # 4-byte big-endian length prefix
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def recv_request(conn: socket.socket) -> dict:
    """Receive a length-prefixed JSON request."""
    # Read 4-byte length prefix
    header = b""
    while len(header) < 4:
        chunk = conn.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("Client disconnected")
        header += chunk

    msg_len = struct.unpack(">I", header)[0]
    if msg_len > MAX_MSG_SIZE:
        raise ValueError(f"Message too large: {msg_len}")

    # Read message body
    data = b""
    while len(data) < msg_len:
        chunk = conn.recv(min(msg_len - len(data), 4096))
        if not chunk:
            raise ConnectionError("Client disconnected")
        data += chunk

    return json.loads(data.decode("utf-8"))


def main() -> int:
    # Remove stale socket
    if Path(SOCKET_PATH).exists():
        try:
            Path(SOCKET_PATH).unlink()
        except OSError:
            print(f"Cannot remove stale socket: {SOCKET_PATH}", file=sys.stderr)
            return 1

    # Load model
    print("Loading embedding model...", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    model = EmbeddingModel()
    model.load()
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s", file=sys.stderr, flush=True)

    # Compute anchor embeddings
    print("Computing anchor embeddings...", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    anchor_store = AnchorStore(model)
    anchor_store.compute()
    anchor_time = time.perf_counter() - t0
    print(f"Anchors computed in {anchor_time:.2f}s", file=sys.stderr, flush=True)

    # Create Unix socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(SOCKET_PATH)
    sock.listen(5)
    sock.settimeout(IDLE_TIMEOUT)

    write_pid()

    # Handle SIGTERM for clean shutdown
    def handle_signal(signum, frame):
        cleanup(sock)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    print(f"Server ready on {SOCKET_PATH} (PID: {os.getpid()})", file=sys.stderr, flush=True)

    try:
        while True:
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                print("Idle timeout reached, shutting down.", file=sys.stderr, flush=True)
                break

            try:
                conn.settimeout(5.0)
                request = recv_request(conn)

                text = request.get("text", "")
                if not text:
                    send_response(conn, {"error": "empty text"})
                else:
                    result = classify_message(text, model, anchor_store)
                    send_response(conn, {
                        "type": result[0],
                        "patterns": result[1],
                        "confidence": result[2],
                        "sentiment": result[3],
                        "decay_days": result[4],
                    })
            except Exception as e:
                try:
                    send_response(conn, {"error": str(e)})
                except OSError:
                    pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
    finally:
        cleanup(sock)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        # Clean up on crash
        try:
            if Path(SOCKET_PATH).exists():
                Path(SOCKET_PATH).unlink()
        except OSError:
            pass
        try:
            if PID_FILE.exists():
                PID_FILE.unlink()
        except OSError:
            pass
        sys.exit(1)
