#!/usr/bin/env python3
"""Build an OpenDLSS-NR model directory from NVIDIA's ``nvngx_dlssnr.dll``.

The DLSS 5 "Neural Rendering" (NR) weights shipped by NVIDIA live inside the
runtime DLL as the PE resource ``10/WEIGHTS_HT/1033``.  That resource is a
little-endian, length-prefixed map of 153 named packed records -- exactly the
153 tensors that OpenDLSS-NR's ``nr::Model`` expects.

This script does two things:

1. Locate and carve the ``WEIGHTS_HT`` resource out of a given DLL.
2. Emit an OpenDLSS-NR model directory::

       <out>/
         manifest.json
         model/stage0.bin ...
         raw/weights_ht.bin        # optional, --keep-raw

The container format and the record framing were recovered by the
``inarikami/dlss5-nr-reverse-engineering`` project; see
``docs/nr-model-access.md`` there.  Nothing here decompresses or decrypts
anything: the runtime stores these bytes verbatim.

Usage::

    python3 tools/extract_dlssnr_model.py \
        --dll /path/to/nvngx_dlssnr.dll \
        --out _downloaded_resources/dlss5nr_model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

# Reference payload from the pinned NBA 2K27 runtime, for cross-checking.
REFERENCE_PAYLOAD_SHA256 = "836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4"
REFERENCE_PAYLOAD_BYTES = 147_695_410

MAX_NAME_BYTES = 4096
MAX_RANK = 32

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")

NAME_RE = re.compile(r"^block(\d+)\.layer(\d+)\.(.+)$")

# OpenDLSS-NR refuses a graph with anything other than 71 blocks.
EXPECTED_BLOCKS = 71


def find_resource(data: bytes) -> int:
    """Return the file offset of the ``WEIGHTS_HT`` resource payload.

    The payload starts with a u64 holding its own total size, immediately
    followed by the u64 length and bytes of the first record name.  Using the
    size plus a plausible name length as the search pattern is robust against
    the resource simply moving between DLL builds.
    """
    pattern = _U64.pack(REFERENCE_PAYLOAD_BYTES) + _U64.pack(len(b"block0.layer0.layer"))
    offset = data.find(pattern)
    if offset < 0:
        raise ValueError(
            "WEIGHTS_HT resource not found: no u64 size header followed by a record name. "
            "Is this an nvngx_dlssnr.dll (build 310.x) with a resource section?"
        )
    return offset


def parse_records(data: bytes) -> list[dict]:
    """Parse the 153-record container, returning names/offsets/sizes."""
    size = len(data)

    def integer(offset: int, layout: struct.Struct, end: int, label: str) -> int:
        if offset < 0 or offset + layout.size > end:
            raise ValueError(f"truncated {label} at byte {offset}")
        return layout.unpack_from(data, offset)[0]

    total = integer(0, _U64, size, "file-size header")
    if total != size:
        raise ValueError(f"file-size header is {total}, actual length is {size}")

    records: list[dict] = []
    seen: set[str] = set()
    offset = _U64.size
    while offset < size:
        name_size = integer(offset, _U64, size, "name length")
        offset += _U64.size
        if not 1 <= name_size <= MAX_NAME_BYTES:
            raise ValueError(f"invalid name length {name_size} at byte {offset - 8}")
        name_end = offset + name_size
        if name_end > size:
            raise ValueError(f"truncated name at byte {offset}")
        name = data[offset:name_end].decode("utf-8", errors="strict")
        if "\0" in name:
            raise ValueError(f"NUL in record name at byte {offset}")
        if name in seen:
            raise ValueError(f"duplicate record name {name!r}")
        seen.add(name)

        outer_size = integer(name_end, _U64, size, "outer record size")
        record_offset = name_end + _U64.size
        if outer_size < 36:
            raise ValueError(f"record {name!r} is shorter than its fixed fields")
        record_end = record_offset + outer_size
        if record_end > size:
            raise ValueError(f"record {name!r} extends past the file")
        inner_size = integer(record_offset, _U64, record_end, "inner record size")
        if inner_size != outer_size:
            raise ValueError(f"record {name!r} has mismatched inner and outer sizes")
        payload_size = integer(record_offset + 8, _U64, record_end, "payload size")
        payload_offset = record_offset + 20
        tail_offset = payload_offset + payload_size
        if tail_offset + 16 > record_end:
            raise ValueError(f"record {name!r} payload overruns its metadata tail")
        rank = integer(tail_offset + 8, _U64, record_end, "rank")
        if rank > MAX_RANK:
            raise ValueError(f"record {name!r} rank {rank} exceeds limit {MAX_RANK}")

        records.append(
            {
                "name": name,
                "payload_offset": payload_offset,
                "payload_bytes": payload_size,
                "rank": rank,
            }
        )
        offset = record_end

    if offset != size:
        raise ValueError(f"record map ended at {offset}, file is {size} bytes")
    return records


def build_model_directory(dll: Path, out: Path, chunk_bytes: int, keep_raw: bool,
                          verify_reference: bool) -> None:
    data = dll.read_bytes()
    resource_offset = find_resource(data)

    declared = _U64.unpack_from(data, resource_offset)[0]
    payload = data[resource_offset : resource_offset + declared]
    if len(payload) != declared:
        raise ValueError(f"resource declares {declared} bytes but only {len(payload)} are present")

    digest = hashlib.sha256(payload).hexdigest()
    print(f"resource offset : {resource_offset}")
    print(f"resource bytes  : {len(payload)}")
    print(f"resource sha256 : {digest}")
    if verify_reference and digest != REFERENCE_PAYLOAD_SHA256:
        print(f"warning: payload differs from the pinned reference {REFERENCE_PAYLOAD_SHA256}",
              file=sys.stderr)

    records = parse_records(payload)
    print(f"records         : {len(records)}")

    # Group tensors into stage files.  Only the (stage, stageOffset, byteLength)
    # triple matters to nr::Model, so any contiguous grouping is valid.  We keep
    # the byte stream identical to the vendor layout (no padding between
    # payloads) and start a new stage once the current one reaches chunk_bytes.
    stages: list[dict] = []
    tensors: list[dict] = []
    stage_blobs: list[bytearray] = []
    stage_index = -1

    def open_stage() -> None:
        nonlocal stage_index
        stage_index += 1
        stages.append({
            "id": f"stage{stage_index}",
            "file": f"stage{stage_index}.bin",
            "packedByteLength": 0,
            "sha256": "",
        })
        stage_blobs.append(bytearray())

    open_stage()
    for record in records:
        if stage_blobs[-1] and len(stage_blobs[-1]) >= chunk_bytes:
            open_stage()
        blob = payload[record["payload_offset"] : record["payload_offset"] + record["payload_bytes"]]
        stage_offset = len(stage_blobs[-1])
        stage_blobs[-1] += blob

        match = NAME_RE.match(record["name"])
        if not match:
            raise ValueError(f"unexpected record name {record['name']!r}")
        tensors.append(
            {
                "name": record["name"],
                "block": int(match.group(1)),
                "layer": int(match.group(2)),
                "parameter": match.group(3),
                "stage": stages[-1]["id"],
                "stageOffset": stage_offset,
                "byteLength": record["payload_bytes"],
            }
        )

    stage_files: list[tuple[str, bytes]] = []
    for stage, blob in zip(stages, stage_blobs):
        stage["packedByteLength"] = len(blob)
        stage["sha256"] = hashlib.sha256(blob).hexdigest()
        stage_files.append((stage["file"], bytes(blob)))

    # Every tensor must lie inside its stage and stages must cover the payload.
    for tensor in tensors:
        stage = next(s for s in stages if s["id"] == tensor["stage"])
        if tensor["stageOffset"] + tensor["byteLength"] > stage["packedByteLength"]:
            raise AssertionError(f"tensor {tensor['name']} exceeds its stage")
    if sum(s["packedByteLength"] for s in stages) != sum(t["byteLength"] for t in tensors):
        raise AssertionError("stage split does not cover the tensor payloads")

    manifest = {
        "totals": {"blockCount": EXPECTED_BLOCKS},
        "stages": stages,
        "tensors": tensors,
    }

    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, blob in stage_files:
        (model_dir / filename).write_bytes(blob)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if keep_raw:
        raw_dir = out / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "weights_ht.bin").write_bytes(payload)

    blocks = {t["block"] for t in tensors}
    print(f"stages          : {len(stages)}")
    print(f"tensors         : {len(tensors)}")
    print(f"blocks          : {len(blocks)} (0..{max(blocks)})")
    print(f"tensor bytes    : {sum(t['byteLength'] for t in tensors)}")
    print(f"wrote           : {out}/manifest.json + {len(stages)} stage files")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dll", type=Path, required=True,
                        help="path to nvngx_dlssnr.dll (build 310.x)")
    parser.add_argument("--out", type=Path, required=True,
                        help="output model directory (manifest.json + model/)")
    parser.add_argument("--chunk-mib", type=int, default=32,
                        help="approximate stage file size in MiB (default 32)")
    parser.add_argument("--keep-raw", action="store_true",
                        help="also write the carved WEIGHTS_HT resource to raw/")
    parser.add_argument("--no-verify-reference", action="store_true",
                        help="do not warn when the payload differs from the pinned reference")
    args = parser.parse_args(argv)

    if not args.dll.is_file():
        print(f"extract_dlssnr_model: no such file: {args.dll}", file=sys.stderr)
        return 1
    try:
        build_model_directory(args.dll, args.out, args.chunk_mib * 1024 * 1024,
                              args.keep_raw, not args.no_verify_reference)
    except (OSError, ValueError, AssertionError) as exc:
        print(f"extract_dlssnr_model: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
