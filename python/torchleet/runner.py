"""Check execution and reporting.

A problem spec is a module in torchleet.problems.<id> exposing:
    ENTRIES : list[str]   names the solver must supply
    CHECKS  : list[callable]  each takes a namespace of entries; plain `assert`
              signals failure, raising Skip signals "cannot run here".

Checks assert *properties* (self-consistency, torch-oracle agreement,
invariants). No reference implementation ships with this package, so a correct
solution that differs from ours in initialization or op order still passes.
"""
from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

PROGRESS = Path.home() / ".torchleet" / "progress.json"

PASS, FAIL, SKIP = "pass", "fail", "skip"
_MARK = {PASS: "✅", FAIL: "❌", SKIP: "–"}


class Skip(Exception):
    """Raised by a check that cannot run in this environment (no GPU, missing extra)."""


def module_name(pid: str) -> str:
    """Problem id -> module name. Ids like '2d-positional-embeddings' need a
    leading underscore because a Python module cannot start with a digit."""
    m = pid.replace("-", "_")
    return f"_{m}" if m[:1].isdigit() else m


def _spec(pid: str):
    try:
        return importlib.import_module(f"torchleet.problems.{module_name(pid)}")
    except ModuleNotFoundError:
        return None


def _require_extras(spec) -> None:
    for pkg in getattr(spec, "EXTRAS", []):
        if importlib.util.find_spec(pkg) is None:
            raise Skip(f"needs `{pkg}` — pip install {pkg}")


def _require_device(spec) -> None:
    if getattr(spec, "DEVICE", "cpu") == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise Skip("requires CUDA — no GPU available here")


def run(pid: str, entries: dict) -> list[tuple[str, str, str]]:
    """Returns [(check_name, status, detail)]."""
    spec = _spec(pid)
    if spec is None:
        return [("grading", SKIP, f"'{pid}' is not auto-graded yet")]

    missing = [e for e in spec.ENTRIES if e not in entries or entries[e] is None]
    if missing:
        return [("entries", FAIL,
                 f"missing {', '.join(missing)} — this problem needs "
                 f"{', '.join(spec.ENTRIES)}")]

    try:
        _require_extras(spec)
        _require_device(spec)
    except Skip as s:
        return [(c.__name__.replace("check_", ""), SKIP, str(s)) for c in spec.CHECKS]

    # Everything the solver passed is exposed, not just required entries, so a
    # check can look for an optional one (e.g. a Triton kernel) and skip without it.
    ns = SimpleNamespace(**entries)
    out = []
    for fn in spec.CHECKS:
        name = fn.__name__.replace("check_", "").replace("_", " ")
        try:
            fn(ns)
            out.append((name, PASS, ""))
        except Skip as s:
            out.append((name, SKIP, str(s)))
        except AssertionError as e:
            out.append((name, FAIL, str(e) or "assertion failed"))
        except Exception as e:  # user code blew up; report, don't crash
            out.append((name, FAIL, f"{type(e).__name__}: {e}"))
    return out


def report(pid: str, results: list[tuple[str, str, str]]) -> bool:
    print(f"\n  torchleet: {pid}")
    for name, status, detail in results:
        line = f"   {_MARK[status]} {name}"
        if detail:
            line += f" — {detail}"
        print(line)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)

    if n_fail:
        print(f"   {n_fail} failing — try hint(\"{pid}\")\n")
        return False
    if not n_pass:
        print("   nothing could be checked here\n")
        return False
    if n_skip:
        # Passed everything runnable, but something could not be verified on this
        # machine. Recorded as partial so `status` never overstates what was proven.
        print(f"   {n_pass} passed, {n_skip} not verified here\n")
        _record(pid, partial=True)
        return True
    print("   all checks passed\n")
    _record(pid)
    return True


def _record(pid: str, partial: bool = False) -> None:
    try:
        PROGRESS.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
        data[pid] = {"passed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "partial": partial}
        PROGRESS.write_text(json.dumps(data, indent=1))
    except OSError:
        pass  # ponytail: progress is a convenience; never fail a pass on disk issues
