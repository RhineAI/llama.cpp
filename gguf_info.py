#!/usr/bin/env python3
"""Print comprehensive information about a GGUF file."""

import sys
import os
import struct
from gguf import GGUFReader
from gguf.constants import GGUFValueType, GGMLQuantizationType


def get_val(f):
    t = f.types[-1] if f.types else None
    if t == GGUFValueType.STRING:
        if len(f.types) > 1:
            return [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
        return bytes(f.parts[f.data[0]]).decode("utf-8", "replace")
    elif t in (
        GGUFValueType.UINT8, GGUFValueType.INT8,
        GGUFValueType.UINT16, GGUFValueType.INT16,
        GGUFValueType.UINT32, GGUFValueType.INT32,
        GGUFValueType.UINT64, GGUFValueType.INT64,
        GGUFValueType.FLOAT32, GGUFValueType.FLOAT64,
        GGUFValueType.BOOL,
    ):
        if len(f.types) > 1:
            vals = []
            for i in f.data:
                v = f.parts[i]
                vals.append(v.tolist()[0] if hasattr(v, "tolist") and v.size == 1 else v)
            return vals
        v = f.parts[f.data[0]]
        return v.tolist()[0] if hasattr(v, "tolist") and v.size == 1 else v
    return f"<type={t}>"


def type_name(t):
    try:
        return GGUFValueType(t).name
    except ValueError:
        return f"UNKNOWN({t})"


def fmt_size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def fmt_val(v, max_len=120):
    s = repr(v) if isinstance(v, list) else str(v)
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.gguf> [--tensors] [--tensor-stats] [--no-meta] [--filter KEY]")
        print()
        print("Options:")
        print("  --tensors       Show full tensor list (default: summary only)")
        print("  --tensor-stats  Show per-tensor byte statistics")
        print("  --no-meta       Skip metadata section")
        print("  --filter KEY    Only show metadata keys containing KEY")
        sys.exit(1)

    path = sys.argv[1]
    show_tensors = "--tensors" in sys.argv
    show_stats = "--tensor-stats" in sys.argv
    no_meta = "--no-meta" in sys.argv
    filter_key = None
    if "--filter" in sys.argv:
        idx = sys.argv.index("--filter")
        if idx + 1 < len(sys.argv):
            filter_key = sys.argv[idx + 1].lower()

    if not os.path.isfile(path):
        print(f"Error: file not found: {path}")
        sys.exit(1)

    file_size = os.path.getsize(path)
    reader = GGUFReader(path)

    # === File Header ===
    print("=" * 70)
    print(f"GGUF File Info: {os.path.basename(path)}")
    print("=" * 70)
    print(f"  Path:          {os.path.abspath(path)}")
    print(f"  File size:     {fmt_size(file_size)} ({file_size:,} bytes)")

    with open(path, "rb") as fp:
        magic = fp.read(4)
        version = struct.unpack("<I", fp.read(4))[0]
        n_tensors = struct.unpack("<Q", fp.read(8))[0]
        n_kv = struct.unpack("<Q", fp.read(8))[0]
    print(f"  Magic:         {magic}")
    print(f"  GGUF version:  {version}")
    print(f"  Tensor count:  {n_tensors}")
    print(f"  KV count:      {n_kv}")
    print()

    # === Metadata ===
    if not no_meta:
        print("=" * 70)
        print(f"Metadata ({len(reader.fields)} fields)")
        print("=" * 70)

        groups = {}
        for key in reader.fields:
            if filter_key and filter_key not in key.lower():
                continue
            prefix = key.split(".")[0] if "." in key else "(top-level)"
            groups.setdefault(prefix, []).append(key)

        for group in sorted(groups.keys()):
            print(f"\n  [{group}]")
            for key in sorted(groups[group]):
                f = reader.fields[key]
                val = get_val(f)
                vtype = type_name(f.types[-1]) if f.types else "?"
                if isinstance(val, list):
                    if len(val) <= 8:
                        val_str = fmt_val(val)
                    else:
                        val_str = f"[{len(val)} items] {fmt_val(val[:5])}..."
                else:
                    val_str = fmt_val(val)
                print(f"    {key} ({vtype}) = {val_str}")
        print()

    # === Tensor Summary ===
    print("=" * 70)
    print(f"Tensor Summary ({len(reader.tensors)} tensors)")
    print("=" * 70)

    type_counts = {}
    type_bytes = {}
    total_bytes = 0
    total_elements = 0
    tensors_by_type = {}

    for t in reader.tensors:
        qtype = GGMLQuantizationType(t.tensor_type).name
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
        type_bytes[qtype] = type_bytes.get(qtype, 0) + t.n_bytes
        total_bytes += t.n_bytes
        n_elem = 1
        for d in t.shape:
            n_elem *= d
        total_elements += n_elem
        tensors_by_type.setdefault(qtype, []).append(t)

    print(f"  Total tensor data: {fmt_size(total_bytes)} ({total_bytes:,} bytes)")
    print(f"  Total elements:    {total_elements:,}")
    overhead = file_size - total_bytes
    print(f"  Metadata overhead: {fmt_size(overhead)} ({overhead / file_size * 100:.1f}%)")
    print()

    print(f"  {'Type':<20} {'Count':>6} {'Size':>12} {'% of total':>10}")
    print(f"  {'-'*20} {'-'*6} {'-'*12} {'-'*10}")
    for qtype in sorted(type_counts.keys()):
        cnt = type_counts[qtype]
        sz = type_bytes[qtype]
        pct = sz / total_bytes * 100 if total_bytes else 0
        print(f"  {qtype:<20} {cnt:>6} {fmt_size(sz):>12} {pct:>9.1f}%")
    print()

    # === Full Tensor List ===
    if show_tensors:
        print("=" * 70)
        print("Full Tensor List")
        print("=" * 70)
        print(f"  {'#':<5} {'Name':<55} {'Type':<12} {'Shape':<25} {'Size':>10}")
        print(f"  {'-'*5} {'-'*55} {'-'*12} {'-'*25} {'-'*10}")
        for i, t in enumerate(reader.tensors):
            qtype = GGMLQuantizationType(t.tensor_type).name
            shape_str = "x".join(str(d) for d in t.shape)
            print(f"  {i:<5} {t.name:<55} {qtype:<12} {shape_str:<25} {fmt_size(t.n_bytes):>10}")
        print()

    # === Per-tensor stats ===
    if show_stats:
        print("=" * 70)
        print("Per-Type Tensor Statistics")
        print("=" * 70)
        for qtype in sorted(tensors_by_type.keys()):
            tensors = tensors_by_type[qtype]
            sizes = [t.n_bytes for t in tensors]
            print(f"\n  {qtype} ({len(tensors)} tensors):")
            print(f"    Total:   {fmt_size(sum(sizes))}")
            print(f"    Min:     {fmt_size(min(sizes))}")
            print(f"    Max:     {fmt_size(max(sizes))}")
            print(f"    Mean:    {fmt_size(sum(sizes) / len(sizes))}")
        print()

    # === Model Architecture Quick Summary ===
    print("=" * 70)
    print("Model Architecture Summary")
    print("=" * 70)
    arch_keys = [
        "general.architecture", "general.name", "general.file_type",
        "general.quantization_version",
    ]
    interesting = ["context_length", "embedding_length", "block_count",
                   "head_count", "head_count_kv", "vocab_size",
                   "feed_forward_length", "rope", "layer_norm"]
    for key in reader.fields:
        for pat in interesting:
            if pat in key and key not in arch_keys:
                arch_keys.append(key)

    for key in arch_keys:
        if key in reader.fields:
            val = get_val(reader.fields[key])
            print(f"  {key} = {fmt_val(val)}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
