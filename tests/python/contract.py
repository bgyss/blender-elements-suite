"""Contract test: the Python client against the real Rust daemon.

Run by `crates/elementsd/tests/python_contract.rs`, which starts the daemon and
passes the endpoint, channel path and graph path as arguments.
"""

import json
import os
import pathlib
import struct
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "addon"))

from blender_elements.bakecmd import bake_command  # noqa: E402
from blender_elements.client import (  # noqa: E402
    CHANNEL_HEADER_BYTES,
    CHANNEL_MAGIC,
    CHANNEL_VERSION,
    PROTOCOL_VERSION,
    ControlClient,
    ElementsError,
    FrameReader,
)


def check_truncated_channel_file_rejected() -> None:
    """A channel file whose header claims more data than the file holds must
    raise, not silently return a short/wrong frame (mmap slicing truncates
    silently rather than raising)."""
    header = bytearray(CHANNEL_HEADER_BYTES)
    struct.pack_into("<I", header, 0, CHANNEL_MAGIC)
    struct.pack_into("<I", header, 4, CHANNEL_VERSION)
    struct.pack_into("<I", header, 8, 8)  # dims[0]
    struct.pack_into("<I", header, 12, 8)  # dims[1]
    struct.pack_into("<I", header, 16, 8)  # dims[2]
    struct.pack_into("<I", header, 20, 1)  # channels
    struct.pack_into("<Q", header, 24, 8 * 8 * 8 * 4)  # buffer_bytes (large)
    struct.pack_into("<Q", header, 32, 0)  # seq

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "truncated.channel"
        path.write_bytes(bytes(header))  # nothing after the header

        try:
            FrameReader(str(path))
        except ElementsError as e:
            assert e.kind == "io", e.kind
        else:
            raise AssertionError("expected ElementsError for a truncated channel file")


def check_bake_command() -> None:
    """The viewport bake must ask for the scene's frame, and read back the file
    the CLI actually names. Blender allows negative frames; the engine does not."""
    args, path = bake_command("elements", "g.elements", "out", 7)
    assert args[args.index("--frames") + 1] == "7", args
    assert path == os.path.join("out", "density.0007.vdb"), path

    args, path = bake_command("elements", "g.elements", "out", -3)
    assert args[args.index("--frames") + 1] == "0", args
    assert path == os.path.join("out", "density.0000.vdb"), path


def check_python_version() -> None:
    """The Core v1 DoD commits to py311; assert it and print it so a run's
    log is self-evidencing about which interpreter actually exercised this
    contract, instead of trusting the caller to have set one up correctly."""
    assert sys.version_info[:2] == (3, 11), (
        f"tests/python/contract.py must run on Python 3.11, got "
        f"{sys.version_info[0]}.{sys.version_info[1]}"
    )
    print(f"python version: {sys.version_info[0]}.{sys.version_info[1]}")


def main(endpoint: str, channel: str, graph: str, stateful_graph: str) -> None:
    check_python_version()
    check_truncated_channel_file_rejected()
    check_bake_command()

    with ControlClient(endpoint) as client:
        ack = client.hello()
        assert ack["protocol_version"] == PROTOCOL_VERSION, ack
        assert ack["adapter"], "adapter name must not be empty"

        loaded = client.load_graph(graph)
        assert loaded["dims"] == [8, 8, 8], loaded
        assert loaded["nodes"] == 2, loaded

        frame = client.render(1)
        assert frame["seq"] == 1, frame
        assert frame["dims"] == [8, 8, 8], frame

        reader = FrameReader(channel)
        try:
            header = reader.header()
            assert header["magic"] == CHANNEL_MAGIC, header
            assert header["dims"] == [8, 8, 8], header

            seq, values = reader.read_latest()
            assert seq == 1, seq
            assert len(values) == 512, len(values)
            assert any(v != 0.0 for v in values), "frame must not be blank"
            assert all(v == v for v in values), "frame contains NaN"

            # A second render must advance the sequence and still read cleanly.
            frame2 = client.render(2)
            assert frame2["seq"] == 2, frame2
            seq2, values2 = reader.read_latest()
            assert seq2 == 2, seq2
            assert len(values2) == 512

            # Loading a graph with different dims reallocates the channel file
            # in place. A reader opened against the old geometry must detect
            # this and raise rather than read stale/out-of-bounds bytes.
            realloc_graph = pathlib.Path(graph).with_name("realloc.elements")
            realloc_graph.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "dims": [4, 4, 4],
                        "nodes": [
                            {"id": 0, "kind": "core.noise_field", "params": {"seed": 3}},
                            {"id": 1, "kind": "core.output", "params": {}},
                        ],
                        "edges": [
                            {"from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0}
                        ],
                        "output": 1,
                    }
                )
            )
            loaded_realloc = client.load_graph(str(realloc_graph))
            assert loaded_realloc["dims"] == [4, 4, 4], loaded_realloc
            # Reallocating the channel resets its sequence counter to 0, so
            # the first render after a resolution change publishes seq 1.
            frame3 = client.render(3)
            assert frame3["seq"] == 1, frame3

            try:
                reader.read_latest()
            except ElementsError as e:
                assert e.kind == "io", e.kind
            else:
                raise AssertionError("expected ElementsError after channel reallocation")

            # The caller must reopen to pick up the new geometry.
            reader.close()
            reader = FrameReader(channel)
            header3 = reader.header()
            assert header3["dims"] == [4, 4, 4], header3
            seq3, values3 = reader.read_latest()
            assert seq3 == 1, seq3
            assert len(values3) == 64, len(values3)
        finally:
            reader.close()

        # A bad graph must raise, not corrupt the session.
        try:
            client.load_graph("/nonexistent/graph.elements")
        except ElementsError as e:
            assert e.kind == "io", e.kind
        else:
            raise AssertionError("expected ElementsError for a missing file")

        # The session must still work.
        assert client.load_graph(graph)["nodes"] == 2

        # A stateful graph: the frame number must reach the timeline, and a
        # negative Blender frame must clamp rather than fail.
        loaded_stateful = client.load_graph(stateful_graph)
        assert loaded_stateful["dims"] == [4, 4, 4], loaded_stateful
        stateful_reader = FrameReader(channel)
        try:
            for frame in (3, 1, -5):
                client.render(frame)
                _, values = stateful_reader.read_latest()
                want = 0.5 * max(frame, 1) / 24.0
                assert all(abs(v - want) < 1e-6 for v in values), (frame, values[:4])
        finally:
            stateful_reader.close()

    print("python contract ok")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
