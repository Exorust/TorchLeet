"use client";

interface Props {
  line: string;
  type?: "input" | "output" | "ascii" | "error" | "prompt" | "success";
}

export default function TerminalOutput({ line, type = "output" }: Props) {
  switch (type) {
    case "ascii":
      return (
        <div className="whitespace-pre-wrap font-mono text-terminal-accent leading-none">
          {line}
        </div>
      );
    case "error":
      return (
        <div className="whitespace-pre-wrap font-mono text-terminal-error">
          {line}
        </div>
      );
    case "input":
      return (
        <div className="whitespace-pre-wrap font-mono">
          <span className="text-terminal-prompt">torchleet:~$ </span>
          <span className="text-terminal-text">{line}</span>
        </div>
      );
    case "prompt":
      return (
        <div className="whitespace-pre-wrap font-mono text-terminal-dim">
          {line}
        </div>
      );
    case "success":
      return (
        <div className="whitespace-pre-wrap font-mono text-terminal-success">
          {line}
        </div>
      );
    case "output":
    default:
      return (
        <div className="whitespace-pre-wrap font-mono text-terminal-text">
          {line}
        </div>
      );
  }
}
