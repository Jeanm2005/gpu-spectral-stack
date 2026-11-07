"""tools/loader_profile.py: the GPU busy-time arithmetic behind the idle percentages."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "loader_profile", ROOT / "tools" / "loader_profile.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_busy_time_is_the_union_of_kernels_and_copies(tmp_path):
    events = [
        {"cat": "kernel", "ts": 0, "dur": 10},
        {"cat": "gpu_memcpy", "ts": 5, "dur": 10},  # overlaps the first: busy 0-15
        {"cat": "cpu_op", "ts": 20, "dur": 100},  # host work: not GPU time
        {"cat": "kernel", "ts": 40, "dur": 10},  # gap 15-40 is idle
        {"cat": "gpu_memset", "ts": 45, "dur": 2},  # inside the previous kernel
    ]
    trace = tmp_path / "t.json"
    trace.write_text(json.dumps({"traceEvents": events}))
    busy, span = load_tool().gpu_busy(trace)
    assert (busy, span) == (25, 50)


def test_no_gpu_activity(tmp_path):
    trace = tmp_path / "t.json"
    trace.write_text(json.dumps({"traceEvents": [{"cat": "cpu_op", "ts": 0, "dur": 5}]}))
    assert load_tool().gpu_busy(trace) == (0.0, 0.0)
