"""TorchLeet: check your solutions to the PyTorch interview problems.

    from torchleet import check
    check("kv-cache", KVCache=KVCache, CachedAttention=CachedAttention)
    check("top-k-sampling", top_k_sample)      # single-entry problems

Checks verify properties of your implementation (self-consistency, agreement
with a torch built-in, invariants). They do not diff your output against a
stored reference, so a correct solution that differs from ours still passes.
"""
from __future__ import annotations

import json

from .runner import PROGRESS, Skip, report, run, _spec

__all__ = ["check", "hint", "status", "Skip"]


def check(problem_id: str, *args, **entries) -> bool:
    """Grade an implementation. Positional args bind to the problem's entries in
    order, so single-entry problems can be called as check("id", my_fn)."""
    spec = _spec(problem_id)
    if spec is not None and args:
        names = [n for n in spec.ENTRIES if n not in entries]
        if len(args) > len(names):
            raise TypeError(
                f"{problem_id} takes {len(spec.ENTRIES)} entr"
                f"{'y' if len(spec.ENTRIES) == 1 else 'ies'}: {', '.join(spec.ENTRIES)}")
        entries.update(dict(zip(names, args)))
    elif args:
        raise TypeError(f"'{problem_id}' is not auto-graded yet — pass entries by name")
    return report(problem_id, run(problem_id, entries))


def hint(problem_id: str, n: int | None = None) -> None:
    """Print the next hint, or hint n (1-based). Hints escalate."""
    spec = _spec(problem_id)
    hints = list(getattr(spec, "HINTS", []) or [])
    if not hints:
        print(f"  no hints for '{problem_id}' yet")
        return
    if n is None:
        for i, h in enumerate(hints, 1):
            print(f"  hint {i}/{len(hints)}: {h}")
            break
        return
    if not 1 <= n <= len(hints):
        print(f"  '{problem_id}' has {len(hints)} hints")
        return
    print(f"  hint {n}/{len(hints)}: {hints[n - 1]}")


def status() -> None:
    """Show which problems you've passed on this machine."""
    if not PROGRESS.exists():
        print("  no problems passed yet — run check(...) on one")
        return
    data = json.loads(PROGRESS.read_text())
    full = [p for p, v in data.items() if not v.get("partial")]
    part = [p for p, v in data.items() if v.get("partial")]
    print(f"\n  torchleet — {len(full)} passed"
          + (f", {len(part)} partial" if part else ""))
    for pid in sorted(full):
        print(f"   ✅ {pid}")
    for pid in sorted(part):
        print(f"   ◐ {pid}  (some checks not verifiable on this machine)")
    print()
