#!/usr/bin/env python3
"""Patch GAM so the per-step history length H is drawn identically on every rank (idempotent).

GAM draws H per batch from `random` independently on each rank, so ranks do unequal work and every
gradient all-reduce waits for the slowest rank (measured 2026-09-03: 49% of GPU time waiting in
NCCL). Seeding the draw with (global_seed, train_steps) keeps the H curriculum and makes the draw
identical across ranks. Applied by setup.sh; usage: gam_sync_H.py <gam_root>
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
loss_py = root / "src/robot/losses/unified_loss.py"
train_py = root / "src/train_robot.py"

s = loss_py.read_text()
if "rng=None" not in s:
    old = ('def sample_H(H_choices: List[int], H_weights: Optional[List[float]] = None) -> int:\n'
           '    """Variable-H curriculum: pick H per batch from choices."""\n'
           '    import random\n'
           '    if H_weights is None:\n'
           '        return random.choice(H_choices)\n'
           '    return random.choices(H_choices, weights=H_weights, k=1)[0]\n')
    new = ('def sample_H(H_choices: List[int], H_weights: Optional[List[float]] = None, rng=None) -> int:\n'
           '    """Variable-H curriculum: pick H per batch from choices (rng: feasibility patch, rank-synchronized)."""\n'
           '    import random\n'
           '    r = rng if rng is not None else random\n'
           '    if H_weights is None:\n'
           '        return r.choice(H_choices)\n'
           '    return r.choices(H_choices, weights=H_weights, k=1)[0]\n')
    assert s.count(old) == 1, "sample_H anchor not found"
    loss_py.write_text(s.replace(old, new))
    print("patched", loss_py)
else:
    print("already patched", loss_py)

t = train_py.read_text()
if "rank-synchronized H" not in t:
    old = "                H = sample_H(unified_H_choices, unified_H_weights)\n"
    new = ("                # feasibility patch: rank-synchronized H draw (same H on every rank each step)\n"
           "                H = sample_H(unified_H_choices, unified_H_weights,\n"
           "                             rng=__import__('random').Random(int(training_cfg.get('global_seed', 42)) * 1000003 + int(train_steps)))\n")
    assert t.count(old) == 1, "sample_H call anchor not found"
    train_py.write_text(t.replace(old, new))
    print("patched", train_py)
else:
    print("already patched", train_py)
