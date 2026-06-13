# TorchLeet MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server that turns any AI assistant into an interactive PyTorch interview coach using the [TorchLeet](https://github.com/Exorust/TorchLeet) problem set.

90 real PyTorch problems from ML/AI interviews at Google, Meta, Anthropic, and more — with progressive hints, learning paths, and company-specific prep plans.

## Quick Start

### Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "torchleet": {
      "command": "npx",
      "args": ["-y", "torchleet-mcp"],
      "env": {
        "TORCHLEET_ROOT": "/path/to/your/TorchLeet/clone"
      }
    }
  }
}
```

### Claude Code

```bash
claude mcp add torchleet -- npx -y torchleet-mcp
```

Set `TORCHLEET_ROOT` in your environment to enable notebook-based hints:

```bash
export TORCHLEET_ROOT=/path/to/your/TorchLeet/clone
```

### Cursor / VS Code

Add to your MCP settings following your editor's MCP configuration format, using `npx -y torchleet-mcp` as the command.

## What It Does

### Tools

| Tool | Description |
|------|-------------|
| `list_questions` | Browse and filter questions by set, difficulty, company, category, or learning track |
| `get_question` | Get full details for a question including Colab links |
| `get_learning_path` | Structured learning paths: LLM path, basics, or advanced |
| `get_hint` | Progressive hints (3 levels) — never reveals the full solution |
| `check_solution` | Verification checklist for submitted code (does not execute) |
| `get_company_prep` | Prioritized study plan for a specific company |
| `get_prerequisites` | Find what to learn before tackling a hard problem |

### Prompts

| Prompt | Description |
|--------|-------------|
| `torchleet-tutor` | Interactive study mode with Socratic questioning |
| `torchleet-interview-prep` | Simulates a technical interview at a specific company |
| `torchleet-review` | Code review focused on PyTorch best practices |
| `torchleet-explain` | Deep-dive concept explanations with math and intuition |

### Resources

| Resource | Description |
|----------|-------------|
| `torchleet://questions` | Full question catalog as JSON |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TORCHLEET_ROOT` | No | Path to your TorchLeet repo clone. Enables notebook-based hints with problem statements, function signatures, and TODO guidance. Without it, the server still works using built-in descriptions. |

## Example Usage

After adding the MCP server, try these in your AI assistant:

- "I'm preparing for an Anthropic interview, help me study"
- "Show me the LLM learning path"
- "Give me a hint for v3-14 (DPO Loss)"
- "List all expert-level questions"
- "What should I learn before tackling FlashAttention?"

Select the **torchleet-tutor** prompt for the best guided learning experience.

## Development

```bash
git clone https://github.com/Exorust/TorchLeet.git
cd TorchLeet/mcp-server
npm install
npm run build
```

Test locally:

```bash
TORCHLEET_ROOT=.. node dist/src/index.js
```

## License

MIT
