# Owner-only setup

Three things live in account settings rather than in files, so they need you.

## 1. GitHub topics + description (2 minutes, highest SEO value)

Currently **0 topics** and a 3-word description, so TorchLeet appears on no
`github.com/topics/*` page — and those pages rank in Google.

```bash
gh repo edit Exorust/TorchLeet \
  --description "LeetCode for PyTorch — 65 ML/AI interview problems from real interviews at Google, Meta, Anthropic. Jupyter notebooks, an auto-grader, and an MCP AI tutor." \
  --add-topic pytorch \
  --add-topic interview-questions \
  --add-topic interview-preparation \
  --add-topic machine-learning-interview \
  --add-topic deep-learning \
  --add-topic llm \
  --add-topic transformers \
  --add-topic from-scratch \
  --add-topic leetcode \
  --add-topic coding-interviews \
  --add-topic triton \
  --add-topic rlhf
```

Also enable Discussions (one checkbox in Settings, or):

```bash
gh api -X PATCH repos/Exorust/TorchLeet -f has_discussions=true
```

## 2. PyPI trusted publishing (unblocks `pip install torchleet`)

See `python/PUBLISHING.md`. Short version: add a pending publisher at
https://pypi.org/manage/account/publishing/ for project `torchleet`, owner
`Exorust`, repo `TorchLeet`, workflow `publish.yml`, environment `pypi`; create a
`pypi` environment in GitHub Settings → Environments; then `git tag py-v0.1.0 &&
git push origin py-v0.1.0`.

**Until this is done, the `pip install torchleet` line in 65 notebooks and on
every problem page does not resolve.**

## 3. Hugging Face Space (optional)

`space/` is ready to push (Gradio app + generated `problems.json`).

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create torchleet --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<your-username>/torchleet /tmp/tl-space
cp space/* /tmp/tl-space/ && cd /tmp/tl-space && git add -A \
  && git commit -m "TorchLeet space" && git push
```

Re-sync after any problem change with `python3 scripts/generate.py json && cp problems.json space/`.

## Also worth your time

`SOURCES.md` currently reads **0 reported / 130 inferred**. Every company tag is
marked `inferred` because no dated first-hand report backs it. If you personally
remember which tags came from real candidate conversations, changing those to
`confidence = "reported"` in the relevant `problem.toml` is the single strongest
thing you can do for the launch narrative — and only you can do it.
