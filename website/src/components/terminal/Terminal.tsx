"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useApp } from "@/context/AppContext";
import { ASCII_ART, WELCOME_MESSAGE } from "@/lib/constants";
import { useTypewriter } from "@/components/terminal/useTypewriter";
import { parseCommand } from "@/components/terminal/CommandParser";
import TerminalOutput from "@/components/terminal/TerminalOutput";
import TerminalInput from "@/components/terminal/TerminalInput";
import SidePanel from "@/components/terminal/SidePanel";
import type { Question } from "@/data/questions";

interface TerminalLine {
  text: string;
  type: "input" | "output" | "ascii" | "error" | "prompt";
}

export default function Terminal() {
  const { setMode } = useApp();
  const [lines, setLines] = useState<TerminalLine[]>([]);
  const [introComplete, setIntroComplete] = useState(false);
  const [sidePanelQuestion, setSidePanelQuestion] = useState<Question | null>(
    null,
  );
  const scrollRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputFocusRef = useRef<HTMLDivElement>(null);

  // Build the intro text for the typewriter
  const asciiLines = ASCII_ART.split("\n");
  const welcomeLines = WELCOME_MESSAGE.split("\n");

  // Typewriter on the full ASCII block
  const { displayedText: asciiDisplayed, isComplete: asciiDone } =
    useTypewriter(ASCII_ART, 2, 300);

  // After ASCII is done, show welcome
  const { displayedText: welcomeDisplayed, isComplete: welcomeDone } =
    useTypewriter(WELCOME_MESSAGE, 15, asciiDone ? 400 : 999999);

  // Mark intro complete when both are done
  useEffect(() => {
    if (asciiDone && welcomeDone) {
      setIntroComplete(true);
    }
  }, [asciiDone, welcomeDone]);

  // Auto-scroll on any change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines, asciiDisplayed, welcomeDisplayed]);

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
      const outputLines: TerminalLine[] = result.lines.map((l) => ({
        text: l,
        type: "output" as const,
      }));

      setLines([...newLines, ...outputLines]);
    },
    [lines, setMode],
  );

  // Click anywhere to focus input
  const handleContainerClick = useCallback(() => {
    // Find and focus the input element
    const input = containerRef.current?.querySelector("input");
    if (input) input.focus();
  }, []);

  return (
    <div
      ref={containerRef}
      className="w-screen h-screen bg-terminal-bg overflow-hidden flex flex-col"
      onClick={handleContainerClick}
    >
      {/* Scrollable output area */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto terminal-scrollbar px-4 pt-4 pb-2"
      >
        {/* Intro: ASCII art via typewriter */}
        {asciiDisplayed && (
          <div className="whitespace-pre font-mono text-terminal-accent leading-none text-xs sm:text-sm mb-4">
            {asciiDisplayed}
          </div>
        )}

        {/* Intro: Welcome message via typewriter */}
        {asciiDone && welcomeDisplayed && (
          <div className="whitespace-pre-wrap font-mono text-terminal-text mb-2">
            {welcomeDisplayed}
          </div>
        )}

        {/* Post-intro blank line separator */}
        {introComplete && lines.length > 0 && <div className="h-2" />}

        {/* Scrollback lines from commands */}
        {introComplete &&
          lines.map((line, i) => (
            <TerminalOutput key={i} line={line.text} type={line.type} />
          ))}
      </div>

      {/* Fixed input at bottom */}
      <TerminalInput onSubmit={handleCommand} disabled={!introComplete} />

      {/* Side panel for question details */}
      <SidePanel
        question={sidePanelQuestion}
        onClose={() => setSidePanelQuestion(null)}
      />
    </div>
  );
}
