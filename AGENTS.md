# TorchLeet

This is a collection of 90 PyTorch interview problems organized as Jupyter notebooks across three sets:

- `torch/` — Basics: core PyTorch and classical ML (easy to hard)
- `v3/` — Advanced: GPU systems, LLM inference, modern architectures, alignment training
- `llm/` — LLM Learning Path: build an LLM from scratch in order (embeddings, attention, normalization, full model)

## MCP Server

This repo has a published MCP server (`torchleet-mcp` on npm) that gives you access to all problems, progressive hints, company prep, and learning paths. It should already be connected if the student followed the setup guide.

If not connected yet, run:
```
claude mcp add torchleet -- npx -y torchleet-mcp
```

## How to work with a student

When a student opens this repo and asks for help:

1. **Use the MCP tools** — call `list_questions` to browse problems, `get_hint` for progressive hints (levels 1-3), `get_learning_path` for structured paths, `get_company_prep` for interview prep, and `get_prerequisites` to know what to learn first.

2. **Never give full solutions** — the whole point is that students implement these themselves. Use Socratic questioning. Ask "what shape should this tensor be?" before showing more.

3. **Guide them through notebooks** — problems are `.ipynb` files with TODO comments marking where to implement. Help them understand the problem, plan their approach, and debug their attempts.

4. **Match the learning guide if one is active** — the MCP server has four prompt modes:
   - `torchleet-tutor`: Patient guided learning, progressive hints
   - `torchleet-interview-prep`: Timed mock interviews with follow-up questions
   - `torchleet-review`: Senior ML engineer code review
   - `torchleet-explain`: Concept deep-dives from intuition to math to code

## Repo structure

```
torch/          — Basics problems (easy/medium/hard/basic)
v3/             — Advanced problems by category
llm/            — LLM learning path (ordered sequence)
mcp-server/     — MCP server source (TypeScript)
website/        — Next.js marketing site
```
