#!/usr/bin/env python3
"""Build the exact-scale dictionary index for the Q8 arena runtime."""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from gguf import GGUFReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = []
    reader = GGUFReader(str(args.model))
    for tensor in reader.tensors:
        if tensor.tensor_type.name != "Q8_0":
            continue
        if len(tensor.shape) != 2 or int(tensor.shape[0]) % 32:
            raise ValueError(f"unsupported Q8 tensor shape: {tensor.name}")
        scales = tensor.data.reshape(-1, 34)[:, :2].copy().view("<u2").ravel()
        table = np.unique(scales)
        if not 1 <= len(table) <= 2048:
            raise ValueError(f"{tensor.name}: {len(table)} scales exceed the runtime's 2048-entry limit")
        record = dict(name=tensor.name, shape=tensor.shape.tolist(), kind="q8_dictionary",
                      bits=max(1, int(len(table) - 1).bit_length()), table_count=len(table))
        records.append((record, table.astype("<u2").tobytes()))
    if not records:
        raise ValueError("the GGUF contains no supported Q8_0 matrices")
    manifest = dict(version=1, source=args.model.name, source_bytes=args.model.stat().st_size,
                    tensors=[])
    with args.output.open("xb") as out:
        out.write(b"Q8INDEX1")
        for record, table in records:
            out.write(b"\0" * (-out.tell() % 64))
            record["table_offset"] = out.tell()
            out.write(table)
            manifest["tensors"].append(record)
        encoded = json.dumps(manifest, separators=(",", ":")).encode()
        out.write(encoded)
        out.write(struct.pack("<Q8s", len(encoded), b"Q8INDEX1"))
    # Validate the actual artifact, including every dictionary and its offsets.
    with args.output.open("rb") as check:
        for record, table in records:
            check.seek(record["table_offset"])
            assert check.read(record["table_count"] * 2) == table
        check.seek(-16, 2)
        size, magic = struct.unpack("<Q8s", check.read(16))
        assert magic == b"Q8INDEX1"
        check.seek(-16 - size, 2)
        assert json.loads(check.read(size)) == manifest
    print(json.dumps(dict(path=str(args.output), bytes=args.output.stat().st_size,
                          dictionaries=len(records), verified=True), indent=2))


if __name__ == "__main__":
    main()
