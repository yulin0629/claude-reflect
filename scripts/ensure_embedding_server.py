#!/usr/bin/env python3
"""Ensure the embedding server daemon is running. SessionStart hook.

Called at the start of each Claude Code session to pre-warm the model.
If the server is already running, this is a no-op (~1ms).
If not, starts it in the background (~2s for model load).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))


def main() -> int:
    try:
        from daemon_client import ensure_server_running, _is_server_running
    except ImportError:
        # Dependencies not installed — skip silently
        return 0

    if _is_server_running():
        return 0

    success = ensure_server_running()
    if success:
        print("Embedding server started (multilingual pattern detection ready)")
    # Don't print failure — silent degradation to no embedding detection

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Never block session start
        sys.exit(0)
