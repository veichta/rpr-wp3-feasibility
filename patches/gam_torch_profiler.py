#!/usr/bin/env python3
"""Patch GAM's train_robot.py with an env-gated torch.profiler window (idempotent).

DA3_TORCH_PROFILE=<start>:<end> profiles micro-steps [start, end) on rank 0 and writes
<experiment_dir>/torch_profile/{trace_rank0.json, kernels_rank0.txt}. Applied by setup.sh.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("src/train_robot.py")
src = path.read_text()
MARK = "DA3_TORCH_PROFILE"
if MARK in src:
    print("already patched")
    sys.exit(0)

init_anchor = "        _prof_t_last_step_end = None\n"
init_code = init_anchor + (
    "        # --- feasibility patch: torch.profiler window (env DA3_TORCH_PROFILE=start:end, rank 0) ---\n"
    "        _tp_env = os.environ.get(\"DA3_TORCH_PROFILE\", \"\")\n"
    "        _tp_range = tuple(int(x) for x in _tp_env.split(\":\")) if (_tp_env and rank == 0) else None\n"
    "        _tp_prof = None\n"
)
loop_anchor = "        while True:\n            if stop_training:\n                break\n"
loop_code = loop_anchor + (
    "            if _tp_range is not None:\n"
    "                if _tp_prof is None and train_steps == _tp_range[0]:\n"
    "                    _tp_prof = torch.profiler.profile(\n"
    "                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],\n"
    "                        record_shapes=False, profile_memory=False, with_stack=False,\n"
    "                    )\n"
    "                    _tp_prof.__enter__()\n"
    "                    logger.info(\"[torch_profile] started at step %d\", train_steps)\n"
    "                elif _tp_prof is not None and train_steps >= _tp_range[1]:\n"
    "                    _tp_prof.__exit__(None, None, None)\n"
    "                    _tp_dir = os.path.join(experiment_dir, \"torch_profile\")\n"
    "                    os.makedirs(_tp_dir, exist_ok=True)\n"
    "                    _tp_prof.export_chrome_trace(os.path.join(_tp_dir, f\"trace_rank{rank}.json\"))\n"
    "                    with open(os.path.join(_tp_dir, f\"kernels_rank{rank}.txt\"), \"w\") as _f:\n"
    "                        _f.write(_tp_prof.key_averages().table(sort_by=\"cuda_time_total\", row_limit=80))\n"
    "                        _f.write(\"\\n\\n\")\n"
    "                        _f.write(_tp_prof.key_averages().table(sort_by=\"cpu_time_total\", row_limit=40))\n"
    "                    logger.info(\"[torch_profile] wrote %s (steps %d-%d)\", _tp_dir, _tp_range[0], train_steps)\n"
    "                    _tp_prof = None\n"
    "                    _tp_range = None\n"
)
assert src.count(init_anchor) == 1, "init anchor not unique"
assert src.count(loop_anchor) == 1, "loop anchor not unique"
src = src.replace(init_anchor, init_code).replace(loop_anchor, loop_code)
path.write_text(src)
print("patched", path)
