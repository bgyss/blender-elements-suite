"""Engine client for the Elements add-on.

Standard library only: this module must import and work on the Python that
ships inside Blender 4.2/5.0 (3.11) through 5.2 (3.13) with no wheels
installed, and outside Blender for contract testing.

Two planes mirror the Rust side:
  * control  -- newline-delimited JSON over a Unix socket or Windows named pipe
  * data     -- a memory-mapped file holding two alternating frame buffers
"""

from __future__ import annotations

import array
import json
import mmap
import os
import socket
import struct
import sys

PROTOCOL_VERSION = 2

CHANNEL_MAGIC = 0x4346_4C45  # "ELFC" little-endian
CHANNEL_VERSION = 1
CHANNEL_HEADER_BYTES = 64

_OFF_MAGIC = 0
_OFF_VERSION = 4
_OFF_DIMS = 8
_OFF_CHANNELS = 20
_OFF_BUFFER_BYTES = 24
_OFF_SEQ = 32

MAX_MESSAGE_BYTES = 1024 * 1024


class ElementsError(Exception):
    """An error reported by the engine, or a transport failure."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message


class _PipeSocket:
    """Minimal socket-alike over a Windows named pipe."""

    def __init__(self, endpoint: str) -> None:
        self._f = open(endpoint, "r+b", buffering=0)  # noqa: SIM115 -- lives with the object

    def sendall(self, data: bytes) -> None:
        self._f.write(data)
        self._f.flush()

    def recv(self, size: int) -> bytes:
        return self._f.read(size)

    def close(self) -> None:
        self._f.close()


class ControlClient:
    """Newline-delimited JSON client for the engine's control plane."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._sock = None
        self._buf = b""

    def __enter__(self) -> ControlClient:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        if sys.platform == "win32":
            self._sock = _PipeSocket(self._endpoint)
        else:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self._endpoint)
            except Exception:
                sock.close()
                raise
            self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _send(self, message: dict) -> None:
        if self._sock is None:
            raise ElementsError("io", "not connected")
        self._sock.sendall(json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n")

    def _recv(self) -> dict:
        if self._sock is None:
            raise ElementsError("io", "not connected")
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ElementsError("io", "engine closed the connection")
            self._buf += chunk
            if len(self._buf) > MAX_MESSAGE_BYTES:
                raise ElementsError("io", "engine sent an oversized message")

        line, self._buf = self._buf.split(b"\n", 1)
        try:
            message = json.loads(line.decode("utf-8"))
        except ValueError as e:
            raise ElementsError("io", f"malformed message from engine: {e}") from e

        if message.get("type") == "error":
            raise ElementsError(
                message.get("kind", "io"), message.get("message", "unknown error")
            )
        return message

    def _round_trip(self, message: dict) -> dict:
        self._send(message)
        return self._recv()

    def hello(self) -> dict:
        return self._round_trip({"type": "hello", "protocol_version": PROTOCOL_VERSION})

    def load_graph(self, path: str) -> dict:
        return self._round_trip({"type": "load_graph", "path": os.fspath(path)})

    def render(self, frame: int) -> dict:
        # `Command::Render { frame: u32 }`: Blender allows negative frames, and a
        # negative number would be rejected as a malformed command. The engine
        # clamps anything below the document's start frame to the start frame,
        # so 0 behaves the same.
        return self._round_trip({"type": "render", "frame": max(0, int(frame))})

    def shutdown(self) -> dict:
        return self._round_trip({"type": "shutdown"})


class FrameReader:
    """Reads the latest published frame from the memory-mapped data plane."""

    def __init__(self, path: str) -> None:
        self._map = None
        self._file = None
        self._file = open(path, "rb")  # noqa: SIM115 -- lives with the object
        try:
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self.close()
            raise

        magic = self._u32(_OFF_MAGIC)
        if magic != CHANNEL_MAGIC:
            self.close()
            raise ElementsError("io", f"not an Elements channel (magic {magic:#x})")
        version = self._u32(_OFF_VERSION)
        if version != CHANNEL_VERSION:
            self.close()
            raise ElementsError("io", f"unsupported channel version {version}")

        self._magic = magic
        self._version = version
        self._dims = [self._u32(_OFF_DIMS + i * 4) for i in range(3)]
        self._channels = self._u32(_OFF_CHANNELS)
        self._buffer_bytes = self._u64(_OFF_BUFFER_BYTES)
        self._values = self._dims[0] * self._dims[1] * self._dims[2] * self._channels

        expected = CHANNEL_HEADER_BYTES + 2 * self._buffer_bytes
        actual = len(self._map)
        if actual < expected:
            self.close()
            raise ElementsError(
                "io",
                f"truncated channel file: expected at least {expected} bytes, got {actual}",
            )

    def _u32(self, offset: int) -> int:
        return struct.unpack_from("<I", self._map, offset)[0]

    def _u64(self, offset: int) -> int:
        return struct.unpack_from("<Q", self._map, offset)[0]

    def header(self) -> dict:
        return {
            "magic": self._magic,
            "version": self._version,
            "dims": list(self._dims),
            "channels": self._channels,
            "buffer_bytes": self._buffer_bytes,
            "seq": self._u64(_OFF_SEQ),
        }

    def read_latest(self) -> tuple[int, array.array]:
        """Return (sequence, values). Sequence 0 means nothing is published yet.

        The engine reallocates the channel file in place whenever a newly
        loaded graph has different dims. `dims` and `buffer_bytes` are
        re-read from the header at the start of every call, before any buffer
        offset is computed from them, and compared against what was cached at
        `open`: this mapping's cached geometry would otherwise describe a
        file that no longer exists in that shape, and computing an offset
        from stale `buffer_bytes` risks slicing past the end of a shrunk
        mapping (SIGBUS) or reading the wrong bytes entirely. The header
        itself always lives in the first `CHANNEL_HEADER_BYTES` bytes, which
        `open` already validated, so reading it here is safe even against a
        shrunk file.

        The writer fills buffer `seq % 2` then bumps `seq`, so re-reading
        `seq` after the copy detects a frame that was overwritten mid-read.
        """
        current_dims = [self._u32(_OFF_DIMS + i * 4) for i in range(3)]
        current_buffer_bytes = self._u64(_OFF_BUFFER_BYTES)
        if current_dims != self._dims or current_buffer_bytes != self._buffer_bytes:
            raise ElementsError(
                "io",
                "the channel at this path was reallocated with different dims; "
                "reopen the FrameReader",
            )

        for _ in range(3):
            before = self._u64(_OFF_SEQ)
            index = 0 if before == 0 else (before - 1) % 2
            start = CHANNEL_HEADER_BYTES + index * self._buffer_bytes

            values = array.array("f")
            values.frombytes(self._map[start : start + self._buffer_bytes])

            if self._u64(_OFF_SEQ) == before:
                return before, values

        raise ElementsError("io", "the engine lapped the reader repeatedly")

    def close(self) -> None:
        if getattr(self, "_map", None) is not None:
            self._map.close()
            self._map = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None
