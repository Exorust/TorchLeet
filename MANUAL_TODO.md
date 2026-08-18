# What only you can do

Everything else in this change set is done and verified. These four need your
accounts or your memory.

---

## 1. PyPI trusted publisher — UNBLOCKS A BROKEN PROMISE (do first)

`pip install torchleet` is printed in **65 notebook badge cells** and on **every
problem page**, and it currently 404s. Nothing else on this list matters as much.

1. Go to https://pypi.org/manage/account/publishing/
2. Under "Add a new pending publisher":
   - PyPI Project Name: `torchleet`
   - Owner: `Exorust`
   - Repository name: `TorchLeet`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. GitHub → Settings → Environments → create an environment named `pypi`
4. Then:
   ```bash
   git tag py-v0.1.0 && git push origin py-v0.1.0
   ```
5. Verify: `pip install torchleet && python -c "from torchleet import status; status()"`

No API token is stored anywhere; this uses OIDC.

---

## 2. GitHub topics + description — 2 minutes, best SEO/effort ratio

The repo has **0 topics** today, so it appears on no `github.com/topics/*` page,
and those pages rank in Google.

```bash
gh repo edit Exorust/TorchLeet \
  --description "LeetCode for PyTorch — 65 ML/AI interview problems from real interviews at Google, Meta, Anthropic. Jupyter notebooks, an auto-grader, and an MCP AI tutor." \
  --add-topic pytorch --add-topic interview-questions \
  --add-topic interview-preparation --add-topic machine-learning-interview \
  --add-topic deep-learning --add-topic llm --add-topic transformers \
  --add-topic from-scratch --add-topic leetcode --add-topic coding-interviews \
  --add-topic triton --add-topic rlhf

gh api -X PATCH repos/Exorust/TorchLeet -f has_discussions=true
```

---

## 3. Hugging Face Space (optional)

`space/` is ready and runs locally today.

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create torchleet --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<your-username>/torchleet /tmp/tl-space
cp space/* /tmp/tl-space/ && cd /tmp/tl-space && git add -A \
  && git commit -m "TorchLeet space" && git push
```

Re-sync after any problem change:
`python3 scripts/generate.py json && cp problems.json space/`

---

## 4. Reclassify the company tags you actually sourced

`SOURCES.md` reads **0 reported / 130 inferred**. Every tag is marked `inferred`
because no dated first-hand report backs it in the data. That is the honest
default, and it is weaker than what the README claims.

For any tag you personally heard from a candidate, edit that problem's
`problem.toml`:

```toml
[[companies]]
name = "Anthropic"
confidence = "reported"   # was: inferred
round = "coding-screen"   # optional
period = "2025-Q3"        # optional - omit rather than guess
```

Then `python3 scripts/generate.py sources`. Do **not** invent dates: an
unreconstructable "2025-Q3" is worse than no date, and it is the first thing a
hostile reader probes.

---

## Decisions worth making (not blocking)

- **13 problems in `PLANNED.toml`** are advertised with no notebook. Either write
  them or drop them from the README.
- **6 problems are intentionally ungraded** (`save-model`, `augmentation`,
  `cuda-amp`, `xai`, `rms-norm`, `create-embeddings-out-of-an-llm`) — they have
  nothing a solver must implement. Fine as-is; just know it is deliberate.
- **`website/package.json` lists `@tailwindcss/oxide-linux-x64-gnu`** as a hard
  dependency. It is Linux-x64 only, so `npm install` fails on any Mac without
  `--force`. Move it to `optionalDependencies`.
- **`vit-mae`, `inference-engine` (tier B) and `speculative-decoding` (tier C)**
  are now fully graded and could be promoted to tier A in their manifests.
