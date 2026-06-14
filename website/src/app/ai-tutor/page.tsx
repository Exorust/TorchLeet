"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { WordRoll, CommunityBadge } from "performative-ui";
import Navbar from "@/components/web/Navbar";
import Footer from "@/components/web/Footer";

type Platform = "claude-code" | "codex" | "claude-desktop" | "cursor";

const platforms: { key: Platform; label: string }[] = [
  { key: "claude-code", label: "Claude Code" },
  { key: "codex", label: "Codex" },
  { key: "claude-desktop", label: "Claude Desktop" },
  { key: "cursor", label: "Cursor / VS Code" },
];

const installConfigs: Record<Platform, { steps: string[]; configLabel: string; config: string }> = {
  "claude-code": {
    steps: ["Run this single command:"],
    configLabel: "Terminal",
    config: `claude mcp add torchleet -- npx -y torchleet-mcp`,
  },
  codex: {
    steps: ["Run this single command:"],
    configLabel: "Terminal",
    config: `codex mcp add torchleet -- npx -y torchleet-mcp`,
  },
  "claude-desktop": {
    steps: [
      "Open Claude Desktop settings",
      "Go to Developer > Edit Config",
      "Paste this into your config file:",
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
      "Add a new server with this config:",
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

const workflows = [
  {
    title: "Guided Study Session",
    prompt: "torchleet-tutor",
    conversation: [
      { role: "user" as const, text: "I want to learn how attention works in transformers" },
      { role: "ai" as const, text: "Great choice! Let's start with the fundamentals. Before we dive into the code, tell me: what do you think \"attention\" means intuitively? What problem does it solve that simpler approaches can't?" },
      { role: "user" as const, text: "It lets the model focus on relevant parts of the input?" },
      { role: "ai" as const, text: "Exactly! Now let's make that concrete. I've pulled up the Scaled Dot-Product Attention problem (#V3-8). Here's your Level 1 hint, just the problem statement and requirements. Try to sketch out the function signature before I show you more." },
    ],
  },
  {
    title: "Interview Prep for Anthropic",
    prompt: "torchleet-interview-prep",
    conversation: [
      { role: "user" as const, text: "I'm interviewing at Anthropic next week, help me prepare" },
      { role: "ai" as const, text: "Let's simulate a real interview. I've found 8 problems tagged with Anthropic, spanning DPO Loss, GRPO, KV Cache, and more. We'll start with a 20-minute timed problem. Ready?" },
      { role: "user" as const, text: "Yes, let's go!" },
      { role: "ai" as const, text: "Your first problem: Implement DPO Loss from scratch. You need to compute the Direct Preference Optimization loss given policy and reference model log-probabilities. Timer starts now. What's your approach?" },
    ],
  },
  {
    title: "Code Review",
    prompt: "torchleet-review",
    conversation: [
      { role: "user" as const, text: "Can you review my FlashAttention implementation?" },
      { role: "ai" as const, text: "I'll review this as a senior ML engineer. Three things I notice right away:\n\n1. **Numerical stability**: your softmax isn't using the log-sum-exp trick\n2. **Memory**: you're materializing the full N×N attention matrix, which defeats the purpose of FlashAttention\n3. **Tiling**: the block size should be tuned to SRAM capacity\n\nLet's fix #2 first since it's the core idea..." },
    ],
  },
];

const prompts = [
  {
    name: "torchleet-tutor",
    label: "Tutor Mode",
    desc: "Guides you through questions",
    color: "text-emerald-600",
    bg: "bg-emerald-50",
    border: "border-emerald-200",
  },
  {
    name: "torchleet-interview-prep",
    label: "Interview Prep",
    desc: "Grills you through timed technical interviews",
    color: "text-blue-600",
    bg: "bg-blue-50",
    border: "border-blue-200",
  },
  {
    name: "torchleet-review",
    label: "Code Review",
    desc: "Reviews the code you've written",
    color: "text-amber-600",
    bg: "bg-amber-50",
    border: "border-amber-200",
  },
  {
    name: "torchleet-explain",
    label: "Concept Deep-Dive",
    desc: "Starts off easy and builds it up for you",
    color: "text-purple-600",
    bg: "bg-purple-50",
    border: "border-purple-200",
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

function Section({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-50px" }}
      transition={{ duration: 0.5 }}
      className={className}
    >
      {children}
    </motion.section>
  );
}

export default function AITutorPage() {
  const [platform, setPlatform] = useState<Platform>("claude-code");
  const instructions = installConfigs[platform];

  return (
    <div className="min-h-screen bg-lavender-50">
      <Navbar />
      <main className="max-w-4xl mx-auto px-4 pt-28 pb-16 space-y-20">
        {/* Hero */}
        <motion.div
          className="space-y-6"
          initial="hidden"
          animate="visible"
          variants={{ visible: { transition: { staggerChildren: 0.12 } } }}
        >
          <motion.div variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0 } }}>
            <Link
              href="/"
              className="inline-flex items-center gap-1.5 text-sm text-foreground/50 hover:text-foreground/80 transition-colors"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              Back to TorchLeet
            </Link>
          </motion.div>

          <div>
            <motion.div
              variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0 } }}
              className="flex items-center gap-3 mb-3"
            >
              <motion.span
                animate={{ scale: [1, 1.08, 1] }}
                transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
                className="text-xs font-bold uppercase tracking-wider bg-lavender-600 text-white px-2.5 py-1 rounded-full"
              >
                New
              </motion.span>
              <span className="text-xs text-foreground/40 font-mono">torchleet-mcp</span>
            </motion.div>
            <motion.h1
              variants={{ hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } }}
              className="text-4xl md:text-5xl font-medium text-lavender-600 leading-tight"
            >
              <WordRoll
                words={["Claude", "Codex", "Cursor", "Hermes"]}
                intervalMs={2400}
                gradient
              />{" "}
              teaches you PyTorch
            </motion.h1>
            <motion.p
              variants={{ hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } }}
              className="text-lg text-foreground/60 leading-relaxed max-w-2xl mt-4"
            >
              Turn any AI assistant into your PyTorch interview coach.
              75 problems, progressive hints, company prep, and learning paths.
              It won&apos;t spoil the answers either.
            </motion.p>
          </div>

          <motion.div
            variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0 } }}
            className="flex items-center gap-4 text-sm"
          >
            <a
              href="https://www.npmjs.com/package/torchleet-mcp"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-foreground/50 hover:text-foreground/70 transition font-medium"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M0 7.334v8h6.666v1.332H12v-1.332h12v-8H0zm6.666 6.664H5.334v-4H3.999v4H1.335V8.667h5.331v5.331zm4 0H8.001V8.667h5.334v5.332h-2.669v-.001zm12.001 0h-1.33v-4h-1.336v4h-1.335v-4h-1.33v4h-2.671V8.667h8.002v5.331zM10.665 10H12v2.667h-1.335V10z"/></svg>
              npm
            </a>
            <a
              href="https://github.com/Exorust/TorchLeet/tree/main/mcp-server"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-foreground/50 hover:text-foreground/70 transition font-medium"
            >
              <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
              Source
            </a>
            <a
              href="https://modelcontextprotocol.io"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-foreground/50 hover:text-foreground/70 transition font-medium"
            >
              What is MCP?
            </a>
          </motion.div>
        </motion.div>

        {/* How it works */}
        <Section>
          <h2 className="text-2xl font-medium text-foreground mb-6">How It Works</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              { step: "1", title: "Clone the repo", desc: "Get the notebooks so the tutor can read actual problems." },
              { step: "2", title: "Connect the tutor", desc: "One command. npx handles the rest." },
              { step: "3", title: "Start learning", desc: "Pick a guide and ask naturally. It finds problems, gives hints, and checks your work." },
            ].map((item, i) => (
              <motion.div
                key={item.step}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.15, duration: 0.4 }}
              >
                <div className="p-6 h-full rounded-2xl border border-gray-200 bg-white/60 backdrop-blur-lg shadow-sm hover:shadow-md hover:border-lavender-200 transition-all">
                  <div className="font-mono text-2xl font-bold text-lavender-600/30 mb-2">{item.step}</div>
                  <div className="font-semibold text-sm text-foreground mb-1">{item.title}</div>
                  <div className="text-xs text-foreground/50">{item.desc}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </Section>

        {/* Clone the repo */}
        <Section>
          <h2 className="text-2xl font-medium text-foreground mb-2">Clone the Repo</h2>
          <p className="text-sm text-foreground/50 mb-6">
            You&apos;ll work through problems as Jupyter notebooks. Clone first so the AI tutor can read the actual problem files.
          </p>

          <div className="bg-white/60 backdrop-blur-lg rounded-2xl border border-white/50 p-6 shadow-sm">
            <div className="relative">
              <div className="text-xs font-mono text-foreground/40 mb-1.5">Terminal</div>
              <pre className="bg-[#1a1625] text-[#e2e0ea] rounded-xl p-4 pr-20 text-sm font-mono overflow-x-auto leading-relaxed">
{`git clone https://github.com/Exorust/TorchLeet.git
cd TorchLeet`}
              </pre>
              <CopyButton text="git clone https://github.com/Exorust/TorchLeet.git && cd TorchLeet" />
            </div>
          </div>
        </Section>

        {/* Connect the AI Tutor */}
        <Section>
          <h2 className="text-2xl font-medium text-foreground mb-2">Connect the AI Tutor</h2>
          <p className="text-sm text-foreground/50 mb-6">
            Run this from inside the TorchLeet folder. <code className="font-mono bg-lavender-50 px-1.5 py-0.5 rounded text-lavender-700">npx</code> handles the rest.
          </p>

          <div className="flex flex-wrap gap-2 mb-4">
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
                    layoutId="install-platform"
                    className="absolute inset-0 bg-lavender-600/5 rounded-xl"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.3 }}
                  />
                )}
                <span className="relative">{p.label}</span>
              </button>
            ))}
          </div>

          <AnimatePresence mode="wait">
            <motion.div
              key={platform}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
              className="bg-white/60 backdrop-blur-lg rounded-2xl border border-white/50 p-6 shadow-sm"
            >
              <ol className="list-decimal list-inside space-y-1.5 text-sm text-foreground/70 mb-4">
                {instructions.steps.map((step, i) => (
                  <li key={i}>{step}</li>
                ))}
              </ol>

              <div className="relative">
                <div className="text-xs font-mono text-foreground/40 mb-1.5">{instructions.configLabel}</div>
                <pre className="bg-[#1a1625] text-[#e2e0ea] rounded-xl p-4 pr-20 text-sm font-mono overflow-x-auto leading-relaxed">
                  {instructions.config}
                </pre>
                <CopyButton text={instructions.config} />
              </div>
            </motion.div>
          </AnimatePresence>
        </Section>

        {/* Prompts */}
        <Section>
          <h2 className="text-2xl font-medium text-foreground mb-6">Learning Guides</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {prompts.map((prompt, i) => (
              <motion.div
                key={prompt.name}
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.05 }}
              >
                <div className={`p-6 rounded-2xl border ${prompt.border} ${prompt.bg}/30 bg-white/60 backdrop-blur-lg shadow-sm hover:shadow-md transition-all`}>
                  <div className={`font-semibold text-base ${prompt.color}`}>{prompt.label}</div>
                  <div className="text-sm text-foreground/60 mt-2 leading-relaxed">{prompt.desc}</div>
                  <div className="font-mono text-xs text-foreground/30 mt-3">{prompt.name}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </Section>

        {/* Example workflows */}
        <Section>
          <h2 className="text-2xl font-medium text-foreground mb-6">Example Workflows</h2>

          <div className="space-y-8">
            {workflows.map((wf, wi) => (
              <motion.div
                key={wi}
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="bg-white/60 backdrop-blur-lg rounded-2xl border border-white/50 overflow-hidden"
              >
                <div className="px-6 py-4 border-b border-white/50 flex items-center gap-3">
                  <span className="font-semibold text-sm text-foreground">{wf.title}</span>
                  <span className="font-mono text-xs text-lavender-600 bg-lavender-50 px-2 py-0.5 rounded-full">
                    {wf.prompt}
                  </span>
                </div>
                <motion.div
                  className="p-6 space-y-3"
                  initial="hidden"
                  whileInView="visible"
                  viewport={{ once: true }}
                  variants={{ visible: { transition: { staggerChildren: 0.1, delayChildren: 0.2 } } }}
                >
                  {wf.conversation.map((msg, mi) => (
                    <motion.div
                      key={mi}
                      variants={{
                        hidden: { opacity: 0, x: msg.role === "user" ? -12 : 12 },
                        visible: { opacity: 1, x: 0 },
                      }}
                      className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                      <div
                        className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                          msg.role === "user"
                            ? "bg-lavender-600 text-white rounded-br-md"
                            : "bg-white border border-gray-200 text-foreground rounded-bl-md"
                        }`}
                      >
                        {msg.role === "ai" && (
                          <div className="text-xs font-medium text-lavender-500 mb-1">TorchLeet</div>
                        )}
                        {msg.text.split("\n").map((line, li) => (
                          <p key={li} className={li > 0 ? "mt-2" : ""}>{line}</p>
                        ))}
                      </div>
                    </motion.div>
                  ))}
                </motion.div>
              </motion.div>
            ))}
          </div>
        </Section>

        {/* Community */}
        <Section className="flex flex-wrap justify-center gap-4">
          <CommunityBadge
            href="https://github.com/Exorust/TorchLeet"
            target="_blank"
            rel="noopener noreferrer"
            iconNode={
              <svg viewBox="0 0 16 16" width="20" height="20" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
            }
            title="GitHub"
            subtitle="Star & contribute"
          />
          <CommunityBadge
            href="https://www.npmjs.com/package/torchleet-mcp"
            target="_blank"
            rel="noopener noreferrer"
            iconNode={
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M0 7.334v8h6.666v1.332H12v-1.332h12v-8H0zm6.666 6.664H5.334v-4H3.999v4H1.335V8.667h5.331v5.331zm4 0H8.001V8.667h5.334v5.332h-2.669v-.001zm12.001 0h-1.33v-4h-1.336v4h-1.335v-4h-1.33v4h-2.671V8.667h8.002v5.331zM10.665 10H12v2.667h-1.335V10z"/></svg>
            }
            title="npm"
            subtitle="Install torchleet-mcp"
          />
          <CommunityBadge
            href="https://discord.gg/torchleet"
            target="_blank"
            rel="noopener noreferrer"
            iconNode={
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.095 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.095 2.157 2.42 0 1.333-.947 2.418-2.157 2.418z"/></svg>
            }
            title="Discord"
            subtitle="Join the community"
          />
        </Section>

      </main>
      <Footer />
    </div>
  );
}
