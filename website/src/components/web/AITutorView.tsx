"use client";

import { useState } from "react";
import { motion } from "framer-motion";

type Platform = "claude-code" | "codex" | "claude-desktop" | "cursor";

const platforms: { key: Platform; label: string; icon: string }[] = [
  { key: "claude-code", label: "Claude Code", icon: "terminal" },
  { key: "codex", label: "Codex", icon: "terminal" },
  { key: "claude-desktop", label: "Claude Desktop", icon: "desktop" },
  { key: "cursor", label: "Cursor / VS Code", icon: "code" },
];

const installInstructions: Record<
  Platform,
  { steps: string[]; config: string; configLabel: string }
> = {
  "claude-code": {
    steps: ["Run this single command — no install needed:"],
    configLabel: "Terminal",
    config: `claude mcp add torchleet -- npx -y torchleet-mcp`,
  },
  codex: {
    steps: ["Run this single command — no install needed:"],
    configLabel: "Terminal",
    config: `codex mcp add torchleet -- npx -y torchleet-mcp`,
  },
  "claude-desktop": {
    steps: [
      "Open Claude Desktop settings",
      "Go to Developer > Edit Config",
      "Paste this into your config file — no install needed, npx handles it:",
    ],
    configLabel: "claude_desktop_config.json",
    config: `{
  "mcpServers": {
    "torchleet": {
      "command": "npx",
      "args": ["-y", "torchleet-mcp"]
    }
  }
}`,
  },
  cursor: {
    steps: [
      "Open Settings > MCP Servers",
      "Add a new server — no install needed, npx handles it:",
    ],
    configLabel: "MCP Server Config",
    config: `{
  "mcpServers": {
    "torchleet": {
      "command": "npx",
      "args": ["-y", "torchleet-mcp"]
    }
  }
}`,
  },
};

const prompts = [
  {
    name: "torchleet-tutor",
    label: "Tutor Mode",
    desc: "Socratic questioning with progressive hints. Never gives full solutions — makes you earn it.",
    color: "text-emerald-600",
    bg: "bg-emerald-50",
    border: "border-emerald-200",
  },
  {
    name: "torchleet-interview-prep",
    label: "Interview Prep",
    desc: "Simulates a timed technical interview at a specific company. Follow-up questions included.",
    color: "text-blue-600",
    bg: "bg-blue-50",
    border: "border-blue-200",
  },
  {
    name: "torchleet-review",
    label: "Code Review",
    desc: "Senior ML engineer reviews your solution for correctness, efficiency, and PyTorch idioms.",
    color: "text-amber-600",
    bg: "bg-amber-50",
    border: "border-amber-200",
  },
  {
    name: "torchleet-explain",
    label: "Concept Deep-Dive",
    desc: "Theory, math, intuition, and production context for any ML concept.",
    color: "text-purple-600",
    bg: "bg-purple-50",
    border: "border-purple-200",
  },
];

const tools = [
  {
    name: "list_questions",
    desc: "Browse and filter by set, difficulty, company, or track",
  },
  {
    name: "get_hint",
    desc: "Progressive hints (3 levels) — never reveals the full solution",
  },
  {
    name: "get_learning_path",
    desc: "Structured paths: LLM from scratch, basics, or advanced",
  },
  {
    name: "get_company_prep",
    desc: "Prioritized study plan for your target company",
  },
  {
    name: "get_prerequisites",
    desc: "What to learn before tackling a hard problem",
  },
  {
    name: "check_solution",
    desc: "Verification checklist for your code (without running it)",
  },
];

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <button
      onClick={() => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }}
      className="absolute top-3 right-3 px-2.5 py-1 rounded-md text-xs font-medium bg-white/80 hover:bg-white border border-white/60 text-foreground/60 hover:text-foreground transition-all"
    >
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}

export default function AITutorView() {
  const [platform, setPlatform] = useState<Platform>("claude-code");
  const instructions = installInstructions[platform];

  return (
    <div className="space-y-10">
      {/* Header */}
      <div>
        <div className="uppercase text-xs tracking-[2px] font-semibold text-lavender-600 mb-1">
          AI-Guided Learning
        </div>
        <h2 className="text-2xl font-medium text-foreground">
          Set Up Your AI Tutor
        </h2>
        <p className="text-sm text-foreground/50 mt-2 max-w-2xl">
          One command. No install. Turn any AI assistant into an interactive
          PyTorch coach with access to all 90 problems, progressive hints,
          company prep plans, and learning paths — while enforcing a
          no-spoilers teaching style.
        </p>
      </div>

      {/* Step 1: Install */}
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="font-mono text-sm px-3 py-1 rounded-full bg-lavender-100 text-lavender-700 font-medium">
            Step 1 — Install
          </div>
        </div>

        {/* Platform selector */}
        <div className="flex flex-wrap gap-2">
          {platforms.map((p) => (
            <button
              key={p.key}
              onClick={() => setPlatform(p.key)}
              className={`relative rounded-xl px-4 py-2.5 text-sm font-medium transition-all border ${
                platform === p.key
                  ? "border-lavender-600 bg-white text-lavender-700 shadow-sm"
                  : "border-white/50 bg-white/40 text-foreground/70 hover:bg-white/70"
              }`}
            >
              {platform === p.key && (
                <motion.div
                  layoutId="platform-active"
                  className="absolute inset-0 bg-lavender-600/5 rounded-xl"
                  transition={{ type: "spring", bounce: 0.2, duration: 0.3 }}
                />
              )}
              <span className="relative">{p.label}</span>
            </button>
          ))}
        </div>

        {/* Instructions */}
        <div className="bg-white/60 backdrop-blur-lg rounded-2xl border border-white/50 p-6 shadow-sm">
          <ol className="list-decimal list-inside space-y-1.5 text-sm text-foreground/70 mb-4">
            {instructions.steps.map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ol>

          <div className="relative">
            <div className="text-xs font-mono text-foreground/40 mb-1.5">
              {instructions.configLabel}
            </div>
            <pre className="bg-[#1a1625] text-[#e2e0ea] rounded-xl p-4 pr-20 text-sm font-mono overflow-x-auto leading-relaxed">
              {instructions.config}
            </pre>
            <CopyButton text={instructions.config} />
          </div>

          <p className="text-xs text-foreground/40 mt-3">
            That&apos;s it — <code className="font-mono bg-lavender-50 px-1 rounded">npx</code> downloads
            and runs it automatically. For richer hints from the actual notebooks, optionally clone the repo and
            set <code className="font-mono bg-lavender-50 px-1 rounded">TORCHLEET_ROOT=/path/to/TorchLeet</code> in
            your env.
          </p>
        </div>
      </div>

      {/* Step 2: Choose a prompt */}
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="font-mono text-sm px-3 py-1 rounded-full bg-lavender-100 text-lavender-700 font-medium">
            Step 2 — Choose a Prompt
          </div>
          <div className="text-xs text-foreground/40">
            Prompts control how the AI teaches you
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {prompts.map((prompt, i) => (
            <motion.div
              key={prompt.name}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className={`rounded-2xl border ${prompt.border} ${prompt.bg}/30 p-5`}
            >
              <div className={`font-semibold text-sm ${prompt.color}`}>
                {prompt.label}
              </div>
              <div className="text-xs text-foreground/60 mt-1">
                {prompt.desc}
              </div>
              <div className="font-mono text-xs text-foreground/30 mt-2">
                {prompt.name}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Step 3: Start learning */}
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="font-mono text-sm px-3 py-1 rounded-full bg-lavender-100 text-lavender-700 font-medium">
            Step 3 — Start Learning
          </div>
          <div className="text-xs text-foreground/40">
            Try these prompts after setup
          </div>
        </div>

        <div className="bg-white/60 backdrop-blur-lg rounded-2xl border border-white/50 p-6 shadow-sm space-y-3">
          {[
            "I'm preparing for an Anthropic interview, help me study",
            "Show me the LLM learning path",
            "Give me a hint for the DPO Loss problem",
            "What should I learn before tackling FlashAttention?",
          ].map((example, i) => (
            <div
              key={i}
              className="flex items-start gap-3 text-sm"
            >
              <span className="text-lavender-600 font-mono shrink-0">{">"}</span>
              <span className="text-foreground/70 italic">
                &ldquo;{example}&rdquo;
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Available tools reference */}
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="font-mono text-sm px-3 py-1 rounded-full bg-lavender-100 text-lavender-700 font-medium">
            Available Tools
          </div>
          <div className="text-xs text-foreground/40">
            What your AI assistant can do
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
          {tools.map((tool) => (
            <div
              key={tool.name}
              className="bg-white/40 rounded-xl border border-white/50 px-4 py-3"
            >
              <div className="font-mono text-xs text-lavender-600 font-medium">
                {tool.name}
              </div>
              <div className="text-xs text-foreground/50 mt-0.5">
                {tool.desc}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* npm badge */}
      <div className="text-xs text-foreground/40 flex items-center gap-2">
        <a
          href="https://www.npmjs.com/package/torchleet-mcp"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-foreground/60 underline"
        >
          torchleet-mcp on npm
        </a>
        <span>&middot;</span>
        <a
          href="https://github.com/Exorust/TorchLeet/tree/main/mcp-server"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-foreground/60 underline"
        >
          Source on GitHub
        </a>
      </div>
    </div>
  );
}
