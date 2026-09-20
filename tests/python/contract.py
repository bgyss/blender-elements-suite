"""Contract test: the Python client against the real Rust daemon.

Run by `crates/elementsd/tests/python_contract.rs`, which starts the daemon and
passes the endpoint, channel path and graph path as arguments.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "addon"))

from blender_elements.client import (  # noqa: E402
    CHANNEL_MAGIC,
    PROTOCOL_VERSION,
    ControlClient,
    ElementsError,
    FrameReader,
)


def main(endpoint: str, channel: str, graph: str) -> None:
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

    print("python contract ok")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
