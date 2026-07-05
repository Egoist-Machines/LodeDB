"""Memory-bandwidth roofline for a batch-1 MiniLM forward pass.

The megakernel idea in issue #67 keeps activations on-chip across all six encoder
layers so they never round-trip through global memory. That helps only to the extent
activation traffic dominates; a batch-1 encoder must still stream every *weight*
through the ALUs once, and no fusion can avoid that read. This module measures the
irreducible weight traffic directly from the ONNX graph and turns it into a
per-device latency floor, which bounds the best speedup any megakernel could deliver.

It reads the actual model initializers (not an analytic guess): each weight tensor is
sized by its shape and dtype, then split into

- **streamed weights** - read in full on every forward pass (all the Linear/LayerNorm
  parameters). This is the irreducible term.
- **gathered embedding tables** - the word / position / token-type lookup tables that
  feed ``Gather`` ops. At batch 1 with ``S`` tokens only ``S`` rows of each are read,
  so the ~47 MB word-embedding table contributes kilobytes, not megabytes.

The floor is ``streamed_bytes / peak_bandwidth``: a perfect memory-bound kernel that
achieves peak bandwidth and keeps every activation on-chip. Real kernels reach
60-80% of peak, so the true floor is higher; this is a deliberately optimistic bound,
which is what makes it a valid *no-go* filter (see the module README). All bandwidth
figures are vendor peak specs, tagged ``reference`` - never measured here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ONNX TensorProto data_type -> bytes per element. Kept explicit rather than pulled
# from onnx.mapping so the accounting does not shift with onnx package versions.
_TENSOR_DTYPE_BYTES: dict[int, int] = {
    1: 4,  # FLOAT
    2: 1,  # UINT8
    3: 1,  # INT8
    4: 2,  # UINT16
    5: 2,  # INT16
    6: 4,  # INT32
    7: 8,  # INT64
    9: 1,  # BOOL
    10: 2,  # FLOAT16
    11: 8,  # DOUBLE
    12: 4,  # UINT32
    13: 8,  # UINT64
    16: 2,  # BFLOAT16
}
_TENSOR_DTYPE_NAME: dict[int, str] = {
    1: "float32",
    10: "float16",
    16: "bfloat16",
    11: "float64",
    6: "int32",
    7: "int64",
}

# Vendor peak memory-bandwidth specs (GB/s, 1 GB = 1e9 bytes). Reference values, never
# measured here; used only to turn a byte budget into a latency floor for context.
DEVICE_PEAK_BANDWIDTH_GBPS: dict[str, float] = {
    "m1_gpu": 68.0,  # Apple M1, unified LPDDR4X (~68.25 GB/s)
    "m1_cpu": 68.0,  # same unified memory the CPU sees
    "a10": 600.0,  # NVIDIA A10, GDDR6
    "l40s": 864.0,  # NVIDIA L40S, GDDR6
    "a100_80g": 2039.0,  # NVIDIA A100 80GB, HBM2e
    "h100_sxm": 3350.0,  # NVIDIA H100 SXM, HBM3
}


@dataclass(frozen=True)
class WeightBudget:
    """Byte accounting for a model's parameters, split by how they are read."""

    onnx_file_bytes: int
    total_param_bytes: int
    streamed_weight_bytes: int
    gathered_table_bytes_full: int
    initializer_count: int
    node_count: int
    dtype_histogram: dict[str, int]
    hidden_size: int | None
    num_layers: int | None
    intermediate_size: int | None
    vocab_size: int | None
    gathered_table_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "onnx_file_bytes": self.onnx_file_bytes,
            "total_param_bytes": self.total_param_bytes,
            "streamed_weight_bytes": self.streamed_weight_bytes,
            "gathered_table_bytes_full": self.gathered_table_bytes_full,
            "initializer_count": self.initializer_count,
            "node_count": self.node_count,
            "dtype_histogram": self.dtype_histogram,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "gathered_table_count": self.gathered_table_count,
        }


@dataclass
class RooflineFloor:
    """Per-device memory-bandwidth floor for one batch-1 forward pass (derived)."""

    seq_len: int
    streamed_bytes_native: int
    streamed_bytes_fp16: int
    gathered_read_bytes: int
    floors_ms: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "provenance": "derived",
            "note": (
                "streamed_bytes / vendor_peak_bandwidth: an optimistic lower bound on "
                "batch-1 forward latency. A megakernel cannot beat this; real kernels "
                "reach 60-80% of peak. fp16 assumes a half-precision kernel."
            ),
            "seq_len": self.seq_len,
            "streamed_bytes_native": self.streamed_bytes_native,
            "streamed_bytes_fp16": self.streamed_bytes_fp16,
            "gathered_read_bytes": self.gathered_read_bytes,
            "floors_ms": self.floors_ms,
        }


def _dtype_bytes(data_type: int) -> int:
    if data_type not in _TENSOR_DTYPE_BYTES:
        raise ValueError(f"unhandled ONNX tensor data_type {data_type}")
    return _TENSOR_DTYPE_BYTES[data_type]


def _numel(dims: list[int]) -> int:
    total = 1
    for dim in dims:
        total *= int(dim)
    return total


def analyze_weight_budget(onnx_path: str | Path) -> WeightBudget:
    """Sizes every initializer in an ONNX model, split into streamed vs gathered.

    Requires the ``onnx`` package (a benchmark-only analysis tool, not a LodeDB
    dependency). Raises ``ImportError`` if it is absent.
    """

    import onnx  # local import: analysis-only, not a runtime dependency

    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path), load_external_data=False)
    graph = model.graph

    # Initializers consumed as the data operand (input[0]) of a Gather are lookup
    # tables: at batch 1 only S rows are read, so they are not streamed in full.
    gather_table_names: set[str] = set()
    for node in graph.node:
        if node.op_type == "Gather" and node.input:
            gather_table_names.add(node.input[0])

    total_param_bytes = 0
    streamed_weight_bytes = 0
    gathered_table_bytes_full = 0
    gathered_table_count = 0
    dtype_histogram: dict[str, int] = {}
    embedding_shape: tuple[int, int] | None = None
    intermediate_size: int | None = None

    for initializer in graph.initializer:
        dims = list(initializer.dims)
        elem_bytes = _dtype_bytes(initializer.data_type)
        size = _numel(dims) * elem_bytes
        total_param_bytes += size
        dtype_name = _TENSOR_DTYPE_NAME.get(initializer.data_type, str(initializer.data_type))
        dtype_histogram[dtype_name] = dtype_histogram.get(dtype_name, 0) + size

        # Only *initializer* Gather operands are embedding tables. The graph also has
        # many Gather ops on shape/index tensors (dynamic-shape handling); those data
        # operands are runtime values, not initializers, so they never land here.
        if initializer.name in gather_table_names and len(dims) == 2:
            gathered_table_bytes_full += size
            gathered_table_count += 1
            # The word-embedding table is the largest 2D gather table; its dim1 is hidden.
            if embedding_shape is None or dims[0] > embedding_shape[0]:
                embedding_shape = (int(dims[0]), int(dims[1]))
        else:
            streamed_weight_bytes += size
            # The FFN up-projection is the widest 2D streamed weight; its larger dim is
            # the intermediate size. (query/key/value/dense are hidden x hidden.)
            if len(dims) == 2:
                wide = max(int(dims[0]), int(dims[1]))
                narrow = min(int(dims[0]), int(dims[1]))
                if embedding_shape and narrow == embedding_shape[1] and (
                    intermediate_size is None or wide > intermediate_size
                ):
                    intermediate_size = wide

    hidden_size = embedding_shape[1] if embedding_shape else None
    vocab_size = embedding_shape[0] if embedding_shape else None
    num_layers = _infer_num_layers(graph)

    return WeightBudget(
        onnx_file_bytes=onnx_path.stat().st_size,
        total_param_bytes=total_param_bytes,
        streamed_weight_bytes=streamed_weight_bytes,
        gathered_table_bytes_full=gathered_table_bytes_full,
        initializer_count=len(graph.initializer),
        node_count=len(graph.node),
        dtype_histogram=dtype_histogram,
        hidden_size=hidden_size,
        num_layers=num_layers,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        gathered_table_count=gathered_table_count,
    )


def _infer_num_layers(graph: object) -> int | None:
    """Best-effort encoder-layer count from initializer/node names (``layer.<n>``)."""

    import re

    pattern = re.compile(r"layer[._](\d+)")
    max_index = -1
    for initializer in graph.initializer:  # type: ignore[attr-defined]
        match = pattern.search(initializer.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    if max_index >= 0:
        return max_index + 1
    return None


def roofline_floor(
    budget: WeightBudget,
    *,
    seq_len: int,
    devices: tuple[str, ...] = ("m1_gpu", "a10", "l40s", "h100_sxm"),
) -> RooflineFloor:
    """Turns a weight budget into per-device latency floors for one batch-1 pass."""

    hidden = budget.hidden_size or 384
    # Native-dtype element size, inferred from the dominant streamed dtype.
    native_elem_bytes = 2 if "float16" in budget.dtype_histogram else 4
    streamed_native = budget.streamed_weight_bytes
    streamed_fp16 = streamed_native // 2 if native_elem_bytes == 4 else streamed_native
    # One S-row read per embedding table (word + position + token-type), native dtype.
    gathered_read = budget.gathered_table_count * seq_len * hidden * native_elem_bytes

    floors: dict[str, dict[str, float]] = {}
    for device in devices:
        bw = DEVICE_PEAK_BANDWIDTH_GBPS.get(device)
        if bw is None:
            continue
        bytes_per_ms = bw * 1e9 / 1e3
        floors[device] = {
            "peak_bandwidth_gbps": bw,
            "floor_ms_native": round((streamed_native + gathered_read) / bytes_per_ms, 5),
            "floor_ms_fp16": round((streamed_fp16 + gathered_read) / bytes_per_ms, 5),
        }

    return RooflineFloor(
        seq_len=seq_len,
        streamed_bytes_native=streamed_native,
        streamed_bytes_fp16=streamed_fp16,
        gathered_read_bytes=gathered_read,
        floors_ms=floors,
    )
