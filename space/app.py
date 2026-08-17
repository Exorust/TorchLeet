"""TorchLeet — browse the problems, then solve them in JupyterLab.

Reads the generated problems.json (written by `scripts/generate.py json`) so it
cannot drift from the repo's manifests.

Layout is two-pane on purpose: finding a problem and reading it are different
jobs, and the previous single-column version made you retype a slug into a
textbox to open anything.
"""
from __future__ import annotations

import json
import os
import pathlib

import gradio as gr

HERE = pathlib.Path(__file__).parent
DATA = json.loads((HERE / "problems.json").read_text())
PROBLEMS: list[dict] = DATA["problems"]
GRADED = sum(1 for p in PROBLEMS if p["graded"])

REPO = "https://github.com/Exorust/TorchLeet/blob/main"
# In the Space, JupyterLab is proxied at /lab by server.py. Locally there is no
# lab, so the button falls back to Colab rather than linking somewhere dead.
LAB_BASE = os.environ.get("TORCHLEET_LAB_BASE", "")

FREQ = {"very-likely": "HOT", "common": "COMMON", "emerging": "NEW"}
TRACK_LABEL = {"basics": "Basics", "llm-path": "LLM path", "advanced": "Advanced"}
DIFFICULTIES = ["basic", "easy", "medium", "hard", "expert"]


def all_tracks() -> list[str]:
    return sorted({t for p in PROBLEMS for t in p["tracks"]})


def all_companies() -> list[str]:
    return sorted({c for p in PROBLEMS for c in p["companies"]})


def sort_key(p: dict):
    """LLM-path problems keep their taught order; everything else by difficulty."""
    order = p.get("llm_path_order")
    return (0, order) if order is not None else (1, DIFFICULTIES.index(p["difficulty"])
                                                 if p["difficulty"] in DIFFICULTIES else 9)


def search(query: str, track: str, difficulty: str, company: str,
           graded_only: bool) -> list[dict]:
    q = (query or "").strip().lower()
    out = []
    for p in PROBLEMS:
        if track != "All" and track not in p["tracks"]:
            continue
        if difficulty != "All" and p["difficulty"] != difficulty:
            continue
        if company != "All" and company not in p["companies"]:
            continue
        if graded_only and not p["graded"]:
            continue
        if q:
            hay = " ".join([p["title"], p["id"], p.get("description", ""),
                            " ".join(p["companies"]), p.get("category", "")]).lower()
            if q not in hay:
                continue
        out.append(p)
    return sorted(out, key=sort_key)


def rows(matches: list[dict]) -> list[list[str]]:
    return [[p["title"], p["difficulty"], "yes" if p["graded"] else ""] for p in matches]


def count_line(matches: list[dict]) -> str:
    n = len(matches)
    if n == len(PROBLEMS):
        return f"**{n} problems** · {GRADED} auto-graded"
    if n == 0:
        return "**No matches.** Try clearing a filter."
    return f"**{n} of {len(PROBLEMS)}** match · {sum(1 for p in matches if p['graded'])} auto-graded"


EMPTY = """### Pick a problem

Search or filter on the left, then click a row.

Every problem is a Jupyter notebook with the parts you implement marked `TODO`.
**Auto-graded** ones can be checked with `torchleet`, which verifies properties of
your implementation rather than diffing it against a stored answer, so a correct
solution written differently from ours still passes.
"""


def detail(p: dict | None) -> tuple[str, str, str]:
    """Returns (markdown, check-snippet, hints-markdown)."""
    if p is None:
        return EMPTY, "", ""

    bits = [f"### {p['title']}", ""]
    meta = [p["difficulty"]]
    if p.get("frequency") in FREQ:
        meta.append(FREQ[p["frequency"]])
    if p.get("category"):
        meta.append(p["category"])
    meta.append("auto-graded" if p["graded"] else "not auto-graded")
    bits += ["`" + "`  `".join(meta) + "`", ""]

    if p.get("description"):
        bits += [p["description"], ""]

    if p["companies"]:
        conf = p.get("company_confidence", {})
        tags = ", ".join(f"{c} ({conf.get(c, 'inferred')})" for c in p["companies"])
        bits += [f"**Asked at** {tags}", ""]

    links = []
    if LAB_BASE:
        links.append(f"### [Solve in JupyterLab →]({LAB_BASE}/tree/{p['question']})")
    links.append(f"### [Open in Colab →]({p['colab']})")
    bits += links + [""]

    nb = [f"[question notebook]({REPO}/{p['question']})"]
    if p.get("solution"):
        nb.append(f"[solution]({REPO}/{p['solution']})")
    bits += ["On GitHub: " + " · ".join(nb)]

    snippet = ""
    if p["graded"]:
        args = "".join(f", {e}" for e in p["entries"])
        snippet = ("pip install torchleet\n\n"
                   "from torchleet import check\n"
                   f'check("{p["id"]}"{args})')

    hints = ""
    if p.get("hints"):
        hints = "\n".join(f"{i}. {h}" for i, h in enumerate(p["hints"], 1))
    return "\n".join(bits), snippet, hints


# Gradio 6 moved theme/css off the Blocks constructor, so they are passed at
# launch() locally and at mount_gradio_app() in the Space (see server.py).
THEME = gr.themes.Base()
CSS = """
    .tl-list table { font-size: 13px; }
    .tl-list td:first-child { font-weight: 500; }
    footer { display: none !important; }
"""

with gr.Blocks(title="TorchLeet") as demo:
    gr.Markdown(
        f"# TorchLeet\n"
        f"{len(PROBLEMS)} PyTorch and LLM interview problems, "
        f"{GRADED} of them auto-graded.  "
        f"[GitHub](https://github.com/Exorust/TorchLeet)")

    matches = gr.State(sorted(PROBLEMS, key=sort_key))

    with gr.Row():
        with gr.Column(scale=2, min_width=320):
            query = gr.Textbox(label="Search", placeholder="attention, kv cache, anthropic…",
                               autofocus=True)
            with gr.Row():
                track = gr.Dropdown(["All"] + [TRACK_LABEL.get(t, t) for t in all_tracks()],
                                    value="All", label="Track")
                difficulty = gr.Dropdown(["All"] + DIFFICULTIES, value="All",
                                         label="Difficulty")
            with gr.Row():
                company = gr.Dropdown(["All"] + all_companies(), value="All",
                                      label="Company")
                graded_only = gr.Checkbox(label="Auto-graded only", value=False)

            count = gr.Markdown(count_line(PROBLEMS))
            table = gr.Dataframe(
                headers=["Problem", "Level", "Graded"],
                datatype=["str", "str", "str"],
                value=rows(sorted(PROBLEMS, key=sort_key)),
                interactive=False, wrap=True, elem_classes="tl-list",
                column_widths=["62%", "22%", "16%"], max_height=560)

        with gr.Column(scale=3, min_width=380):
            body = gr.Markdown(EMPTY)
            snippet = gr.Code(value="", language="python", label="Check your answer",
                              visible=False)
            hint_box = gr.Accordion("Hints", open=False, visible=False)
            with hint_box:
                hints_md = gr.Markdown("")

    def refilter(q, t, d, c, g):
        # Track dropdown shows friendly labels; map back to the stored keys.
        key = next((k for k, v in TRACK_LABEL.items() if v == t), t)
        found = search(q, key, d, c, g)
        return found, rows(found), count_line(found)

    def pick(found, evt: gr.SelectData):
        if not found or evt.index is None:
            return EMPTY, gr.update(visible=False), "", gr.update(visible=False)
        row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if row >= len(found):
            return EMPTY, gr.update(visible=False), "", gr.update(visible=False)
        md, code, hints = detail(found[row])
        return (md,
                gr.update(value=code, visible=bool(code)),
                hints,
                gr.update(visible=bool(hints)))

    controls = [query, track, difficulty, company, graded_only]
    for ctl in controls:
        ctl.change(refilter, controls, [matches, table, count])
    table.select(pick, matches, [body, snippet, hints_md, hint_box])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", theme=THEME, css=CSS,
                server_port=int(os.environ.get("PORT", 7860)))
