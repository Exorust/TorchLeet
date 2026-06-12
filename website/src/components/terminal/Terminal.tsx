"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useApp } from "@/context/AppContext";
import { WELCOME_MESSAGE } from "@/lib/constants";
import { ASCII_ART } from "@/lib/ascii-art";
import { parseCommand } from "@/components/terminal/CommandParser";
import TerminalOutput from "@/components/terminal/TerminalOutput";
import TerminalInput from "@/components/terminal/TerminalInput";
import SidePanel from "@/components/terminal/SidePanel";
import type { Question } from "@/data/questions";

interface TerminalLine {
  text: string;
  type: "input" | "output" | "ascii" | "error" | "prompt" | "success";
}

export default function Terminal() {
  const { mode, setMode } = useApp();
  const [lines, setLines] = useState<TerminalLine[]>([]);
  const [sidePanelQuestion, setSidePanelQuestion] = useState<Question | null>(
    null,
  );
  const scrollRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on any change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines]);

  const handleCommand = useCallback(
    (input: string) => {
      const trimmed = input.trim();

      // Add user input to scrollback
      const newLines: TerminalLine[] = [
        ...lines,
        { text: trimmed || "", type: "input" },
      ];

      if (!trimmed) {
        setLines(newLines);
        return;
      }

      const result = parseCommand(trimmed);

      if (result.action === "clear") {
        setLines([]);
        return;
      }

      if (result.action === "exit") {
        setLines([...newLines, ...result.lines.map((l) => ({ text: l, type: "output" as const }))]);
        // Small delay so user sees the message
        setTimeout(() => setMode("web"), 400);
        return;
      }

      if (result.action === "open" && result.openQuestion) {
        setSidePanelQuestion(result.openQuestion);
      }

      // Add output lines
      const outputLines: TerminalLine[] = result.lines.map((l) => {
        const trimmed = l.trim();
        const isSuccess =
          trimmed.startsWith("[x]") ||
          trimmed.startsWith("Marked complete:") ||
          trimmed.startsWith("Status: [x] Complete");
        return {
          text: l,
          type: isSuccess ? "success" : ("output" as const),
        };
      });

      setLines([...newLines, ...outputLines]);
    },
    [lines, setMode],
  );

  useEffect(() => {
    if (mode === "terminal") {
      const input = containerRef.current?.querySelector("input");
      if (input) input.focus();
    }
  }, [mode]);

  // Click anywhere to focus input
  const handleContainerClick = useCallback(() => {
    // Find and focus the input element
    const input = containerRef.current?.querySelector("input");
    if (input) input.focus();
  }, []);

  return (
    <div
      ref={containerRef}
      className="w-screen h-screen bg-terminal-bg overflow-hidden flex flex-col font-mono"
      onClick={handleContainerClick}
    >
      <style>{`
        @font-face {
          font-family: "IBM_VGA";
          src: url("/fonts/Web437_IBM_VGA_8x16.woff") format("woff");
          font-display: swap;
        }
      `}</style>

      {/* Scrollable output area */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto terminal-scrollbar px-4 pt-4 pb-2"
      >
        {/* ASCII art */}
        <pre
          className="text-terminal-accent mb-4"
          style={{ fontFamily: "IBM_VGA, monospace", fontSize: "16px", lineHeight: "16px" }}
        >
{ASCII_ART}
        </pre>

        {/* Welcome message */}
        <div className="whitespace-pre-wrap font-mono text-terminal-text mb-2">
          {WELCOME_MESSAGE}
        </div>

        {/* Scrollback lines from commands */}
        {lines.map((line, i) => (
          <TerminalOutput key={i} line={line.text} type={line.type} />
        ))}

        {/* Input prompt - inline, right after content */}
        <TerminalInput onSubmit={handleCommand} disabled={false} />
      </div>

      {/* Side panel for question details */}
      <SidePanel
        question={sidePanelQuestion}
        onClose={() => setSidePanelQuestion(null)}
      />
    </div>
  );
}
