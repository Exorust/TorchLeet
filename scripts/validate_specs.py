#!/usr/bin/env python3
"""Run every grader spec against the repo's own _SOLN notebook.

If a spec's checks are wrong or too strict, the canonical solution fails them and
this goes red. That makes it the gate that keeps `check()` honest as specs are
added: a check nobody can pass is worse than no check at all.

Cells are executed in order only until every ENTRY the spec needs is defined, so
training loops, downloads and plots at the bottom of a notebook never run.

Run: python3 scripts/validate_specs.py [problem-id ...]
"""
from __future__ import annotations

import io
import json
import sys
import tomllib
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def solution_namespace(nb_path: Path, entries: list[str]) -> tuple[dict, str | None]:
    """exec notebook code cells until all entries exist. Returns (ns, error)."""
    cells = json.loads(nb_path.read_text())["cells"]
    # Execute inside a real module registered in sys.modules. Decorators that
    # introspect their target - torch.jit.script resolving a helper it calls -
    # need a genuine module to resolve names against; a bare dict is not enough.
    import linecache
    import types
    modname = f"_torchleet_nb_{abs(hash(str(nb_path)))}"
    module = types.ModuleType(modname)
    sys.modules[modname] = module
    ns: dict = module.__dict__
    last_err = None
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if not src.strip():
            continue
        # Compile against a synthetic <...> filename whose text is seeded into
        # linecache, the way IPython does. Using the real .ipynb path instead
        # makes inspect.getsource hand torch.jit.script raw notebook JSON, and
        # `@torch.jit.script` dies with "Expected a single top-level function".
        fname = f"<{nb_path.name} cell {i}>"
        linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
        try:
            with redirect_stdout(io.StringIO()):
                exec(compile(src, fname, "exec"), ns)
        except Exception as e:                      # demo cells may need data we lack
            last_err = f"{type(e).__name__}: {e}"
            # A cell that dies partway still has useful statements after the
            # failure - an import block that trips on torchvision would otherwise
            # lose the `import torch.nn.functional as F` on the next line. Retry
            # statement by statement so the namespace gets everything that works.
            try:
                import ast
                for node in ast.parse(src).body:
                    try:
                        with redirect_stdout(io.StringIO()):
                            exec(compile(ast.Module(body=[node], type_ignores=[]),
                                         fname, "exec"), ns)
                    except Exception:
                        pass
            except SyntaxError:
                pass
        if all(e in ns for e in entries):
            return ns, None
    missing = [e for e in entries if e not in ns]
    return ns, f"never defined {missing} (last cell error: {last_err})"


def main(argv: list[str]) -> int:
    specs = {p.stem.lstrip("_").replace("_", "-"): p
             for p in (ROOT / "python" / "torchleet" / "problems").glob("*.py")
             if p.stem != "__init__"}
    wanted = set(argv) or set(specs)

    manifests = {}
    for mf in ROOT.rglob("problem.toml"):
        if ".git" in mf.parts:
            continue
        m = tomllib.loads(mf.read_text())
        manifests[m["id"]] = (m, mf.parent)

    from torchleet import check

    ok, bad, skipped = [], [], []
    for pid in sorted(wanted):
        if pid not in specs:
            print(f"?? no spec module for '{pid}'")
            continue
        if pid not in manifests:
            bad.append((pid, "no problem.toml with this id"))
            continue
        man, d = manifests[pid]
        sol = man["notebooks"].get("solution") or man["notebooks"]["question"]
        entries = man["grading"]["entries"]

        ns, err = solution_namespace(d / sol, entries)
        if err:
            skipped.append((pid, err))
            continue
        try:
            with redirect_stdout(io.StringIO()) as buf:
                passed = check(pid, **{e: ns[e] for e in entries})
        except Exception as e:
            bad.append((pid, f"grader crashed: {type(e).__name__}: {e}"))
            continue
        if passed:
            ok.append(pid)
        else:
            detail = [l.strip() for l in buf.getvalue().splitlines() if "❌" in l]
            bad.append((pid, "; ".join(detail) or "failed"))

    print("\n" + "=" * 68)
    for pid in ok:
        print(f"  PASS  {pid}")
    for pid, why in skipped:
        print(f"  SKIP  {pid}: {why[:110]}")
    for pid, why in bad:
        print(f"  FAIL  {pid}: {why[:160]}")
    print("=" * 68)
    print(f"{len(ok)} solutions pass their own checks, {len(bad)} fail, {len(skipped)} skipped")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
