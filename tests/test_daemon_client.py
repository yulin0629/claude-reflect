"""Tests for daemon_client.py — Unix socket communication with embedding server.

Uses mocks to avoid requiring a real running server.
"""
import json
import os
import socket
import struct
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "lib"))

from lib.daemon_client import (
    classify_via_daemon,
    ensure_server_running,
    _is_server_running,
    _is_socket_ready,
    stop_server,
    SOCKET_PATH,
    PID_FILE,
)


# =============================================================================
# Tests: _is_server_running
# =============================================================================

class TestIsServerRunning:
    def test_no_pid_file(self, tmp_path):
        with patch("lib.daemon_client.PID_FILE", tmp_path / "nonexistent.pid"):
            assert _is_server_running() is False

    def test_pid_file_with_current_process(self, tmp_path):
        pid_file = tmp_path / "server.pid"
        pid_file.write_text(str(os.getpid()))
        with patch("lib.daemon_client.PID_FILE", pid_file):
            assert _is_server_running() is True

    def test_pid_file_with_dead_process(self, tmp_path):
        pid_file = tmp_path / "server.pid"
        pid_file.write_text("999999999")  # Very unlikely to be a real PID
        with patch("lib.daemon_client.PID_FILE", pid_file):
            result = _is_server_running()
            assert result is False
            # Should clean up stale PID file
            assert not pid_file.exists()

    def test_pid_file_with_invalid_content(self, tmp_path):
        pid_file = tmp_path / "server.pid"
        pid_file.write_text("not-a-number")
        with patch("lib.daemon_client.PID_FILE", pid_file):
            assert _is_server_running() is False


# =============================================================================
# Tests: classify_via_daemon
# =============================================================================

class TestClassifyViaDaemon:
    def test_successful_classification(self):
        """Mock a successful socket exchange."""
        response = {
            "type": "auto",
            "patterns": "embedding:correction",
            "confidence": 0.75,
            "sentiment": "correction",
            "decay_days": 90,
        }

        response_bytes = json.dumps(response).encode("utf-8")
        response_with_header = struct.pack(">I", len(response_bytes)) + response_bytes

        mock_sock = MagicMock()
        mock_sock.recv = MagicMock(side_effect=[
            response_with_header[:4],  # header
            response_with_header[4:],  # body
        ])

        with patch("lib.daemon_client.ensure_server_running", return_value=True), \
             patch("socket.socket") as mock_socket_cls:
            mock_socket_cls.return_value = mock_sock
            result = classify_via_daemon("no, use this instead")

        assert result[0] == "auto"
        assert result[1] == "embedding:correction"
        assert result[2] == 0.75
        assert result[3] == "correction"
        assert result[4] == 90

    def test_server_not_running_returns_fallback(self):
        with patch("lib.daemon_client.ensure_server_running", return_value=False):
            result = classify_via_daemon("test message")

        assert result == (None, "", 0.0, "correction", 90)

    def test_connection_error_returns_fallback(self):
        with patch("lib.daemon_client.ensure_server_running", return_value=True), \
             patch("socket.socket") as mock_socket_cls:
            mock_socket_cls.return_value.connect.side_effect = ConnectionRefusedError()
            result = classify_via_daemon("test message")

        assert result == (None, "", 0.0, "correction", 90)

    def test_timeout_returns_fallback(self):
        with patch("lib.daemon_client.ensure_server_running", return_value=True), \
             patch("socket.socket") as mock_socket_cls:
            mock_socket_cls.return_value.recv.side_effect = socket.timeout()
            result = classify_via_daemon("test message")

        assert result == (None, "", 0.0, "correction", 90)

    def test_server_error_response_returns_fallback(self):
        """Server responds with error JSON."""
        response = {"error": "model not loaded"}
        response_bytes = json.dumps(response).encode("utf-8")
        response_with_header = struct.pack(">I", len(response_bytes)) + response_bytes

        mock_sock = MagicMock()
        mock_sock.recv = MagicMock(side_effect=[
            response_with_header[:4],
            response_with_header[4:],
        ])

        with patch("lib.daemon_client.ensure_server_running", return_value=True), \
             patch("socket.socket") as mock_socket_cls:
            mock_socket_cls.return_value = mock_sock
            result = classify_via_daemon("test")

        assert result == (None, "", 0.0, "correction", 90)

    def test_result_is_always_5_tuple(self):
        """Even on failure, result should be a valid 5-tuple."""
        with patch("lib.daemon_client.ensure_server_running", side_effect=Exception("boom")):
            result = classify_via_daemon("test")

        assert len(result) == 5
        assert result[0] is None


# =============================================================================
# Tests: ensure_server_running
# =============================================================================

class TestEnsureServerRunning:
    def test_already_running(self):
        with patch("lib.daemon_client._is_server_running", return_value=True), \
             patch("lib.daemon_client._is_socket_ready", return_value=True):
            assert ensure_server_running() is True

    def test_no_uv_available(self):
        with patch("lib.daemon_client._is_server_running", return_value=False), \
             patch("lib.daemon_client._is_socket_ready", return_value=False), \
             patch("lib.daemon_client._find_python", return_value=""):
            assert ensure_server_running() is False

    def test_server_script_missing(self, tmp_path):
        with patch("lib.daemon_client._is_server_running", return_value=False), \
             patch("lib.daemon_client._is_socket_ready", return_value=False), \
             patch("lib.daemon_client._find_python", return_value="/usr/bin/uv"), \
             patch("lib.daemon_client._SERVER_SCRIPT", tmp_path / "nonexistent.py"), \
             patch("lib.daemon_client.SOCKET_PATH", "/tmp/test-nonexistent.sock"):
            assert ensure_server_running() is False


# =============================================================================
# Tests: stop_server
# =============================================================================

class TestStopServer:
    def test_no_pid_file(self, tmp_path):
        with patch("lib.daemon_client.PID_FILE", tmp_path / "nonexistent.pid"):
            assert stop_server() is False

    def test_sends_sigterm(self, tmp_path):
        pid_file = tmp_path / "server.pid"
        pid_file.write_text("12345")

        with patch("lib.daemon_client.PID_FILE", pid_file), \
             patch("os.kill") as mock_kill, \
             patch("time.sleep"):
            assert stop_server() is True
            mock_kill.assert_called_once()
