#!/usr/bin/env python3
"""TorchLeet manifest tooling. Manifests (problem.toml) are the source of truth;
questions.ts, README tables and the grader registry are generated from them.

Subcommands:
  seed   one-time: create problem.toml for every problem directory on disk
  check  validate manifests against disk (CI gate)

Reads TOML with stdlib tomllib. Writes it with a small emitter here, because the
only writer is `seed` and pulling in a dependency for one-time output is not worth
it for contributors or CI.
"""
from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TREES = ("torch", "llm", "v3")
# ponytail: '.ipynb_checkpoints', NOT 'checkpoint' — v3/alignment-training/
# gradient-checkpointing/ is a real problem and a looser filter silently drops it.
SKIP = (".git", ".ipynb_checkpoints", "node_modules")

STDLIB_ISH = {"torch", "math", "time", "os", "sys", "json", "random", "typing",
              "dataclasses", "collections", "itertools", "functools", "re", "copy",
              "abc", "warnings", "pathlib", "string", "unicodedata"}


# ---------------------------------------------------------------- notebook io

def notebooks(root: Path = ROOT):
    for p in sorted(root.rglob("*.ipynb")):
        if any(s in p.parts for s in SKIP):
            continue
        yield p


def code_of(nb: Path) -> str:
    try:
        cells = json.loads(nb.read_text())["cells"]
    except Exception:
        return ""
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def first_markdown(nb: Path) -> str:
    try:
        cells = json.loads(nb.read_text())["cells"]
    except Exception:
        return ""
    for c in cells:
        if c["cell_type"] == "markdown":
            return "".join(c["source"])
    return ""


# ------------------------------------------------------------- auto-detection

def detect_entries(question_src: str) -> list[str]:
    """Top-level classes/functions the solver must implement: those whose body
    still contains a TODO / stub marker in the question notebook."""
    out, lines = [], question_src.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^(class|def)\s+(\w+)", line)
        if not m:
            continue
        body = []
        for nxt in lines[i + 1:]:
            if nxt and not nxt[0].isspace() and not nxt.startswith(("#", ")", "]")):
                break
            body.append(nxt)
        blob = "\n".join(body)
        if re.search(r"TODO|NotImplementedError|^\s*\.\.\.\s*$|^\s*pass\s*$", blob, re.M):
            out.append(m.group(2))
    return out


def detect_extras(src: str) -> list[str]:
    mods = set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w]*)", src, re.M))
    return sorted(m for m in mods - STDLIB_ISH if not m.startswith("_"))


def detect_device(src: str) -> str:
    if re.search(r"\btriton\b|@triton\.jit", src):
        return "cuda"
    if re.search(r"\bdist\.init_process_group|FullyShardedDataParallel|\bFSDP\b", src):
        return "cuda"
    return "cpu"


def detect_tier(question_src: str) -> str:
    """Heuristic seed only; humans confirm. Counts real verification signal."""
    n = len(re.findall(r"\bassert\b", question_src)) + \
        len(re.findall(r"allclose", question_src))
    return "A" if n >= 2 else ("B" if n == 1 else "C")


def pair_files(files: list[str]) -> tuple[str | None, str | None]:
    """Map a directory's notebooks to (question, solution) across the four
    conventions that exist in this repo."""
    f = sorted(files)
    us = [x for x in f if "_SOLN" in x]
    hy = [x for x in f if "-SOLN" in x]
    qn = [x for x in f if "-Question" in x]
    if us:
        return (next((x for x in f if x not in us), None), us[0])
    if hy:
        return (next((x for x in f if x not in hy), None), hy[0])
    if qn:  # base filename is the SOLUTION in this convention
        return (qn[0], next((x for x in f if x not in qn), None))
    if len(f) == 1:
        return (f[0], None)
    return (f[0], None)


# ------------------------------------------------------------- toml emitting

def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def emit(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(emit(x) for x in v) + "]"
    if v is None:
        return '""'
    s = str(v)
    if "\n" in s:
        return '"""\n' + s.replace("\\", "\\\\").replace('"""', '\\"""') + '\n"""'
    return '"' + esc(s) + '"'


def write_manifest(path: Path, m: dict) -> None:
    L = []
    for k in ("id", "legacy_ids", "title", "description", "difficulty", "number",
              "set", "category", "tracks"):
        if m.get(k) not in (None, [], ""):
            L.append(f"{k} = {emit(m[k])}")
    L.append("")
    L.append("[notebooks]")
    L.append(f"question = {emit(m['question'])}")
    L.append(f"solution = {emit(m['solution'])}" if m["solution"]
             else "solution = \"\"   # no separate solution file")
    L.append("")
    L.append("[grading]")
    L.append(f"tier = {emit(m['tier'])}        # A=auto-graded B=partial C=not auto-graded (SEEDED, confirm)")
    L.append(f"entries = {emit(m['entries'])}")
    L.append(f"extras = {emit(m['extras'])}")
    L.append(f"device = {emit(m['device'])}")
    if m.get("llm_path"):
        L += ["", "[llm_path]",
              f"order = {m['llm_path']['order']}",
              f"stage = {emit(m['llm_path']['stage'])}"]
    for c in m.get("companies", []):
        L += ["", "[[companies]]", f"name = {emit(c)}",
              'confidence = "inferred"   # inferred | reported']
    path.write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ commands

# Catalog rows that were marked hasNotebook:false while the notebook was in fact
# sitting on disk, unlinked. Reattaching them keeps the legacy id resolvable and
# keeps them off the "planned" list.
FALSE_BLANKS = {"torch/medium/alexnet": "v1-18", "llm/RMS-Norm": "v2-2"}


def slug_of(d: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", Path(d).name.lower()).strip("-")


def cmd_seed() -> int:
    catalog = json.loads(Path("/tmp/catalog.json").read_text())
    by_id = {q["id"]: q for q in catalog}
    by_dir: dict[str, dict] = {}
    for q in catalog:
        if q.get("questionPath"):
            by_dir[str(Path(q["questionPath"]).parent)] = q

    dirs: dict[str, list[str]] = {}
    for nb in notebooks():
        rel = nb.relative_to(ROOT)
        if rel.parts[0] not in TREES:
            continue
        dirs.setdefault(str(rel.parent), []).append(rel.name)

    made = orphans = 0
    for d, files in sorted(dirs.items()):
        entry = by_dir.get(d)
        qf, sf = pair_files(files)
        if entry:
            qf = Path(entry["questionPath"]).name
            sf = Path(entry["solutionPath"]).name if entry.get("solutionPath") else sf
            m = dict(id=slug_of(d), legacy_ids=[entry["id"]], title=entry["title"],
                     description=entry.get("description") or "",
                     difficulty=entry["difficulty"], number=entry.get("number"),
                     set=entry["set"], category=entry.get("category") or "",
                     tracks=entry.get("tracks") or [], companies=entry.get("companies") or [])
            if entry.get("llmPathOrder") is not None:
                m["llm_path"] = {"order": entry["llmPathOrder"],
                                 "stage": entry.get("llmPathStage") or ""}
        else:
            orphans += 1
            blank = by_id.get(FALSE_BLANKS.get(d, ""))
            title = (blank or {}).get("title") or \
                Path(d).name.replace("-", " ").replace("_", " ").title()
            m = dict(id=slug_of(d),
                     legacy_ids=[blank["id"]] if blank else [],
                     title=title,
                     description=(blank or {}).get("description") or "",
                     difficulty=(blank or {}).get("difficulty") or "medium",
                     number=(blank or {}).get("number"),
                     set=(blank or {}).get("set") or "v2",
                     category=(blank or {}).get("category") or "",
                     tracks=(blank or {}).get("tracks") or [],
                     companies=(blank or {}).get("companies") or [])
            print(f"  {'RELINKED false blank' if blank else 'ORPHAN (needs review)'}: {d}")

        qsrc = code_of(ROOT / d / qf) if qf else ""
        both = qsrc + "\n" + (code_of(ROOT / d / sf) if sf else "")
        m.update(question=qf, solution=sf, entries=detect_entries(qsrc),
                 extras=detect_extras(both), device=detect_device(both),
                 tier=detect_tier(qsrc))
        write_manifest(ROOT / d / "problem.toml", m)
        made += 1

    claimed = {by_dir[d]["id"] for d in dirs if d in by_dir} | \
              {v for k, v in FALSE_BLANKS.items() if k in dirs}
    planned = [q for q in catalog
               if not q.get("hasNotebook") and q["id"] not in claimed]
    Path(ROOT / "PLANNED.toml").write_text(
        "# Advertised problems with no notebook yet. Claim one via an issue.\n"
        "# Generated by scripts/generate.py seed; not counted in the problem total.\n\n"
        + "\n".join(
            f'[[planned]]\nid = {emit(q["id"])}\ntitle = {emit(q["title"])}\n'
            f'difficulty = {emit(q["difficulty"])}\n' for q in planned))

    print(f"\nseeded {made} manifests ({orphans} orphan dirs), "
          f"{len(planned)} planned entries -> PLANNED.toml")
    return 0


def _spec_path(pid: str) -> Path | None:
    if not pid:
        return None
    mod = pid.replace("-", "_")
    mod = f"_{mod}" if mod[:1].isdigit() else mod
    f = ROOT / "python" / "torchleet" / "problems" / f"{mod}.py"
    return f if f.exists() else None


def _spec_entries(pid: str) -> list[str] | None:
    """ENTRIES declared by a grader spec, if one exists. Read as text so this
    stays importable without torch installed (CI validates manifests, not torch)."""
    f = _spec_path(pid)
    if f is None:
        return None
    m = re.search(r"^ENTRIES\s*=\s*\[(.*?)\]", f.read_text(), re.M | re.S)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else []


def _spec_hints(pid: str) -> list[str]:
    """HINTS from the grader spec. The spec module is the single source for
    hints; the Space, the website and the MCP server all read them from the
    generated output rather than keeping their own copies."""
    f = _spec_path(pid)
    if f is None:
        return []
    m = re.search(r"^HINTS\s*=\s*\[(.*?)^\]", f.read_text(), re.M | re.S)
    if not m:
        return []
    # Hints are written as adjacent-string literals across lines; join each
    # element's pieces so "a" \n "b" becomes one hint, not two.
    out = []
    for item in re.findall(r'((?:\s*"(?:[^"\\]|\\.)*"\s*)+),', m.group(1) + ","):
        parts = re.findall(r'"((?:[^"\\]|\\.)*)"', item)
        if parts:
            out.append("".join(parts).replace('\\"', '"'))
    return out


def cmd_check() -> int:
    errs, seen, n = [], {}, 0
    for mf in sorted(ROOT.rglob("problem.toml")):
        if any(s in mf.parts for s in SKIP):
            continue
        n += 1
        rel = mf.relative_to(ROOT)
        try:
            m = tomllib.loads(mf.read_text())
        except Exception as e:
            errs.append(f"{rel}: unparseable ({e})")
            continue
        for k in ("id", "title", "difficulty", "notebooks", "grading"):
            if k not in m:
                errs.append(f"{rel}: missing required key '{k}'")
        pid = m.get("id")
        if pid in seen:
            errs.append(f"{rel}: duplicate id '{pid}' (also {seen[pid]})")
        seen[pid] = rel
        q = (m.get("notebooks") or {}).get("question")
        if not q:
            errs.append(f"{rel}: notebooks.question is empty")
        elif not (mf.parent / q).exists():
            errs.append(f"{rel}: notebooks.question '{q}' does not exist")
        s = (m.get("notebooks") or {}).get("solution")
        if s and not (mf.parent / s).exists():
            errs.append(f"{rel}: notebooks.solution '{s}' does not exist")
        grading = m.get("grading") or {}
        if grading.get("tier") not in ("A", "B", "C"):
            errs.append(f"{rel}: grading.tier must be A, B or C")
        # A problem with a grader spec must agree with it about the entry points,
        # or solvers get told to implement one thing while being graded on another.
        want_h = _spec_hints(pid)
        if want_h and (m.get("hints") or []) != want_h:
            errs.append(f"{rel}: hints differ from the grader spec — "
                        f"run `generate.py hints`")
        spec = _spec_entries(pid)
        if spec is not None and sorted(spec) != sorted(grading.get("entries") or []):
            errs.append(f"{rel}: grading.entries {grading.get('entries')} != "
                        f"torchleet.problems ENTRIES {spec}")

    for e in errs:
        print(f"FAIL {e}")
    print(f"\n{n} manifests checked, {len(errs)} problems")
    return 1 if errs else 0


REPO = "Exorust/TorchLeet"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/main"
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
MARK = "<!-- torchleet:colab -->"


def load_all() -> list[tuple[dict, Path]]:
    out = []
    for mf in sorted(ROOT.rglob("problem.toml")):
        if any(s in mf.parts for s in SKIP):
            continue
        out.append((tomllib.loads(mf.read_text()), mf.parent))
    return sorted(out, key=lambda t: (t[0].get("set", ""), t[0].get("number") or 0))


def colab_url(m: dict, d: Path) -> str:
    return f"{COLAB}/{(d.relative_to(ROOT) / m['notebooks']['question']).as_posix()}"


def cmd_badges() -> int:
    """Put an 'Open in Colab' cell at the top of every question notebook.
    Idempotent: the marker cell is replaced, never stacked."""
    n = 0
    for m, d in load_all():
        nb_path = d / m["notebooks"]["question"]
        nb = json.loads(nb_path.read_text())
        cell = {
            "cell_type": "markdown", "metadata": {},
            "source": [f"{MARK}\n",
                       f"[![Open In Colab]({BADGE})]({colab_url(m, d)})\n",
                       "\n",
                       "Check your work: `!pip install torchleet` then "
                       f"`from torchleet import check; check(\"{m['id']}\", ...)`\n"],
        }
        cells = [c for c in nb["cells"]
                 if not (c["cell_type"] == "markdown" and MARK in "".join(c["source"]))]
        nb["cells"] = [cell] + cells
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        n += 1
    print(f"badged {n} question notebooks")
    return 0


def cmd_readme() -> int:
    """Write the generated index to PROBLEMS.md and sync the counts quoted in
    README.md. The README's curated sections (LLM path staging, company
    quick-reference) are hand-written and deliberately left alone."""
    rows = {"basics": [], "llm-path": [], "advanced": []}
    for m, d in load_all():
        graded = "✅" if _spec_entries(m.get("id")) is not None else ""
        freq = {"very-likely": "🔥", "common": "⭐", "emerging": "💡"}.get(
            m.get("frequency", ""), "")
        comps = ", ".join(c["name"] for c in m.get("companies", [])) or ""
        q = (d.relative_to(ROOT) / m["notebooks"]["question"]).as_posix()
        s = m["notebooks"].get("solution")
        links = f"[Q]({q})" + (f" / [S]({(d.relative_to(ROOT) / s).as_posix()})" if s else "")
        row = (f"| {freq} {m['title']} | {m['difficulty']} | {comps} | {graded} | "
               f"[![Colab]({BADGE})]({colab_url(m, d)}) | {links} |")
        for t in m.get("tracks", []) or ["advanced"]:
            if t in rows:
                rows[t].append(row)

    head = ("| Problem | Difficulty | Companies | Graded | Run | Links |\n"
            "|---|---|---|---|---|---|")
    body = []
    for track, title in (("basics", "Basics"), ("llm-path", "LLM Learning Path"),
                         ("advanced", "Advanced")):
        body += [f"\n### {title} ({len(rows[track])})\n", head, *rows[track]]

    total = len(load_all())
    graded = sum(1 for m, _ in load_all() if _spec_entries(m.get("id")) is not None)
    (ROOT / "PROBLEMS.md").write_text(
        f"# All {total} problems\n\n"
        "Generated from the per-problem `problem.toml` manifests — do not edit by hand.\n"
        f"Run `python3 scripts/generate.py readme` to refresh.\n\n"
        f"**{graded} of {total} are auto-graded** (`pip install torchleet`). A ✅ in the\n"
        "Graded column means `check()` will verify your solution.\n"
        + "\n".join(body) + "\n")

    # Keep the numbers the README quotes in step with reality.
    readme = ROOT / "README.md"
    txt = readme.read_text()
    txt = re.sub(r"\*\*\d+ PyTorch problems from real",
                 f"**{total} PyTorch problems from real", txt)
    txt = re.sub(r"access to all \d+ problems", f"access to all {total} problems", txt)
    txt = re.sub(r"^\d+ problems across three tracks:",
                 f"{total} problems across three tracks:", txt, flags=re.M)
    readme.write_text(txt)
    print(f"PROBLEMS.md written ({sum(len(v) for v in rows.values())} rows); "
          f"README counts synced to {total}")
    return 0


def cmd_sources() -> int:
    """Generate SOURCES.md from company attestations."""
    lines = [
        "# Where these problems come from", "",
        "Problems are collected from first-person accounts by candidates who sat the",
        "interviews. Identities and verbatim prompts are deliberately withheld: the",
        "people who shared them are often under NDA, and naming them would be a",
        "problem for them, not for us.",
        "",
        "That means most tags below are marked **inferred** rather than **reported**.",
        "Inferred means we placed the problem with a company from public job posts,",
        "published engineering material, and what the role plainly requires, not from",
        "a dated first-hand report we can point to. We would rather label that",
        "honestly than invent a precision we do not have.",
        "",
        "If you were asked one of these, "
        f"[tell us](https://github.com/{REPO}/issues/new?template=interview-report.yml)"
        " and it becomes a **reported** tag with a real date.",
        "",
    ]
    by_company: dict[str, list[tuple[str, str]]] = {}
    for m, _ in load_all():
        for c in m.get("companies", []):
            note = c.get("confidence", "inferred")
            if c.get("round"):
                note += f", {c['round']}"
            if c.get("period"):
                note += f", {c['period']}"
            by_company.setdefault(c["name"], []).append((m["title"], note))

    counts = {"reported": 0, "inferred": 0}
    for lst in by_company.values():
        for _, note in lst:
            counts["reported" if note.startswith("reported") else "inferred"] += 1
    lines += [f"**{counts['reported']} reported, {counts['inferred']} inferred "
              f"across {len(by_company)} companies.**", ""]
    for comp in sorted(by_company):
        lines.append(f"### {comp}")
        lines += [f"- {t} — _{n}_" for t, n in sorted(by_company[comp])]
        lines.append("")
    (ROOT / "SOURCES.md").write_text("\n".join(lines))
    print(f"SOURCES.md written ({counts['reported']} reported, "
          f"{counts['inferred']} inferred)")
    return 0


def cmd_website() -> int:
    """Emit the generated problem data the site's SEO pages read.

    Written as a separate module rather than replacing the hand-maintained
    questions.ts, which carries curated helpers the existing UI depends on. This
    file is the manifest-derived source for the per-problem pages and sitemap.
    """
    items = []
    for m, d in load_all():
        rel = d.relative_to(ROOT).as_posix()
        q = f"{rel}/{m['notebooks']['question']}"
        s = m["notebooks"].get("solution")
        items.append({
            "slug": m["id"],
            "legacyIds": m.get("legacy_ids", []),
            "title": m["title"],
            "description": m.get("description", ""),
            "difficulty": m["difficulty"],
            "category": m.get("category", ""),
            "tracks": m.get("tracks", []),
            "companies": [c["name"] for c in m.get("companies", [])],
            "companyConfidence": {c["name"]: c.get("confidence", "inferred")
                                  for c in m.get("companies", [])},
            "questionPath": q,
            "solutionPath": f"{rel}/{s}" if s else None,
            "colabUrl": colab_url(m, d),
            "graded": _spec_entries(m.get("id")) is not None,
            "entries": m.get("grading", {}).get("entries", []),
        })

    out = ROOT / "website" / "src" / "data" / "problems.generated.ts"
    body = ",\n".join("  " + json.dumps(i) for i in items)
    out.write_text(
        "// GENERATED by scripts/generate.py website — do not edit.\n"
        "// Source of truth: the per-problem problem.toml manifests.\n\n"
        "export interface GeneratedProblem {\n"
        "  slug: string;\n  legacyIds: string[];\n  title: string;\n"
        "  description: string;\n  difficulty: string;\n  category: string;\n"
        "  tracks: string[];\n  companies: string[];\n"
        "  companyConfidence: Record<string, string>;\n"
        "  questionPath: string;\n  solutionPath: string | null;\n"
        "  colabUrl: string;\n  graded: boolean;\n  entries: string[];\n}\n\n"
        f"export const generatedProblems: GeneratedProblem[] = [\n{body},\n];\n\n"
        "export function getProblemBySlug(slug: string): GeneratedProblem | undefined {\n"
        "  return generatedProblems.find((p) => p.slug === slug);\n}\n\n"
        "/** Legacy ids (v3-12, ...) still resolve, so older links keep working. */\n"
        "export function getProblemByAnyId(id: string): GeneratedProblem | undefined {\n"
        "  return generatedProblems.find((p) => p.slug === id || p.legacyIds.includes(id));\n}\n")
    print(f"website data written: {len(items)} problems -> {out.relative_to(ROOT)}")
    return 0


MINUTES = {"basic": 20, "easy": 30, "medium": 45, "hard": 90, "expert": 120}


def cmd_studyplan() -> int:
    """Generated study plan. Estimates come from difficulty, so they move with
    the catalogue instead of going stale in prose."""
    tracks: dict[str, list] = {}
    for m, _ in load_all():
        for t in m.get("tracks", []) or ["advanced"]:
            tracks.setdefault(t, []).append(m)

    lines = ["# Study plans", "",
             "Generated from the manifests — estimates are per-problem budgets by",
             "difficulty (basic 20m, easy 30m, medium 45m, hard 90m, expert 2h), not",
             "measurements. Treat them as relative effort, not a promise.", ""]
    for track, title in (("basics", "Basics"), ("llm-path", "LLM Learning Path"),
                         ("advanced", "Advanced")):
        ms = tracks.get(track, [])
        if not ms:
            continue
        mins = sum(MINUTES.get(m["difficulty"], 45) for m in ms)
        weeks = max(1, round(mins / 60 / 5))
        lines += [f"## {title}", "",
                  f"**{len(ms)} problems · ~{mins // 60} hours · "
                  f"about {weeks} week{'s' if weeks > 1 else ''} at 5 h/week**", ""]
        for d in ("basic", "easy", "medium", "hard", "expert"):
            sub = [m for m in ms if m["difficulty"] == d]
            if sub:
                lines.append(f"- {len(sub)} {d} (~{sum(MINUTES[d] for _ in sub) // 60}h)")
        lines.append("")
    (ROOT / "STUDYPLAN.md").write_text("\n".join(lines))
    print("STUDYPLAN.md written")
    return 0


def _spec_hints(pid: str) -> list[str] | None:
    """HINTS authored in a grader spec. Parsed as text so this works without torch."""
    if not pid:
        return None
    mod = pid.replace("-", "_")
    mod = f"_{mod}" if mod[:1].isdigit() else mod
    f = ROOT / "python" / "torchleet" / "problems" / f"{mod}.py"
    if not f.exists():
        return None
    # Parse rather than regex: hint strings are routinely split across lines and
    # implicitly concatenated, which a regex silently truncates.
    import ast
    try:
        tree = ast.parse(f.read_text())
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "HINTS" for t in node.targets):
            try:
                return list(ast.literal_eval(node.value))
            except ValueError:
                return []
    return []


def cmd_hints() -> int:
    """Copy each grader spec's HINTS into its manifest.

    Hints are authored next to the checks they describe, so the spec module owns
    them; the manifest carries a generated copy so non-Python consumers (the MCP
    tutor, the site, the Space) read one source instead of inventing their own.
    `check` fails if the two drift.
    """
    n = 0
    for m, d in load_all():
        hints = _spec_hints(m.get("id"))
        if not hints:
            continue
        mf = d / "problem.toml"
        txt = re.sub(r"\n*^# GENERATED from the grader spec.*?^hints = \[.*?^\]\n",
                     "\n", mf.read_text(), flags=re.M | re.S)
        body = "\n".join(f'  "{esc(h)}",' for h in hints)
        block = ("\n# GENERATED from the grader spec — edit "
                 "python/torchleet/problems/*.py, then run: generate.py hints\n"
                 f"hints = [\n{body}\n]\n")
        # Must land BEFORE the first table header: in TOML every key after a
        # `[table]` line belongs to that table, so appending at the end would
        # bury `hints` inside the last [[companies]] entry.
        head = re.search(r"^\[", txt, re.M)
        txt = (txt[:head.start()].rstrip() + "\n" + block + "\n" + txt[head.start():]
               if head else txt.rstrip() + "\n" + block)
        mf.write_text(txt)
        n += 1
    print(f"hints synced into {n} manifests")
    return 0


def cmd_json() -> int:
    """Flat problems.json — consumed by the HF Space and available to any other
    client that should not hand-maintain a second copy of the catalogue."""
    items = []
    for m, d in load_all():
        rel = d.relative_to(ROOT).as_posix()
        items.append({
            "id": m["id"],
            "legacy_ids": m.get("legacy_ids", []),
            "title": m["title"],
            "description": m.get("description", ""),
            "difficulty": m["difficulty"],
            "category": m.get("category", ""),
            "tracks": m.get("tracks", []),
            "companies": [c["name"] for c in m.get("companies", [])],
            "question": f"{rel}/{m['notebooks']['question']}",
            "solution": (f"{rel}/{m['notebooks']['solution']}"
                         if m["notebooks"].get("solution") else None),
            "colab": colab_url(m, d),
            "frequency": m.get("frequency", ""),
            "company_confidence": {c["name"]: c.get("confidence", "inferred")
                                   for c in m.get("companies", [])},
            "llm_path_order": (m.get("llm_path") or {}).get("order"),
            "graded": _spec_entries(m.get("id")) is not None,
            "entries": m.get("grading", {}).get("entries", []),
            "hints": _spec_hints(m.get("id")),
        })
    blob = json.dumps({"count": len(items), "problems": items}, indent=1) + "\n"
    (ROOT / "problems.json").write_text(blob)
    # Write the Space's copy from the same call: a hand-copied snapshot is how it
    # ended up advertising 30 graded when the manifests said 56.
    space = ROOT / "space"
    if space.is_dir():
        (space / "problems.json").write_text(blob)
    graded = sum(1 for i in items if i["graded"])
    hinted = sum(1 for i in items if i["hints"])
    print(f"problems.json written ({len(items)} problems, {graded} graded, "
          f"{hinted} with hints) -> root + space/")
    return 0


def cmd_coverage() -> int:
    """The published grading-coverage number. Derived, never typed by hand."""
    from collections import Counter
    total, graded = 0, 0
    per_tree: dict[str, list[int]] = {}
    tiers: Counter = Counter()
    for mf in sorted(ROOT.rglob("problem.toml")):
        if any(s in mf.parts for s in SKIP):
            continue
        m = tomllib.loads(mf.read_text())
        total += 1
        tree = mf.relative_to(ROOT).parts[0]
        per_tree.setdefault(tree, [0, 0])
        per_tree[tree][0] += 1
        tiers[m.get("grading", {}).get("tier")] += 1
        if _spec_entries(m.get("id")) is not None:
            graded += 1
            per_tree[tree][1] += 1

    print(f"\n  {graded} of {total} problems are auto-graded "
          f"({graded / total * 100:.0f}%)\n")
    for tree, (n, g) in sorted(per_tree.items()):
        print(f"    {tree:7} {g:3}/{n:<3} graded")
    print(f"\n  tiers: " + "  ".join(f"{t}={tiers[t]}" for t in "ABC"))
    print(f"  planned (no notebook yet): "
          f"{(ROOT / 'PLANNED.toml').read_text().count('[[planned]]')}\n")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "seed":
        sys.exit(cmd_seed())
    if cmd == "check":
        sys.exit(cmd_check())
    if cmd == "coverage":
        sys.exit(cmd_coverage())
    if cmd == "badges":
        sys.exit(cmd_badges())
    if cmd == "readme":
        sys.exit(cmd_readme())
    if cmd == "sources":
        sys.exit(cmd_sources())
    if cmd == "website":
        sys.exit(cmd_website())
    if cmd == "json":
        sys.exit(cmd_json())
    if cmd == "studyplan":
        sys.exit(cmd_studyplan())
    if cmd == "hints":
        sys.exit(cmd_hints())
    print(__doc__)
    sys.exit(2)
