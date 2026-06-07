"use client";

import { useState, useRef, useCallback, type KeyboardEvent } from "react";
import { useTerminalHistory } from "@/hooks/useTerminalHistory";

interface Props {
  onSubmit: (command: string) => void;
  disabled?: boolean;
}

export default function TerminalInput({ onSubmit, disabled = false }: Props) {
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { addToHistory, navigateHistory } = useTerminalHistory();

  const handleSubmit = useCallback(() => {
    const trimmed = value.trim();
    if (trimmed) {
      addToHistory(trimmed);
    }
    onSubmit(value);
    setValue("");
  }, [value, onSubmit, addToHistory]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") {
        e.preventDefault();
        handleSubmit();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setValue(navigateHistory("up", value));
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setValue(navigateHistory("down", value));
      }
    },
    [handleSubmit, navigateHistory, value],
  );

  const focusInput = useCallback(() => {
    inputRef.current?.focus();
  }, []);

  if (disabled) return null;

  return (
    <div
      ref={containerRef}
      className="flex items-center px-4 py-2 font-mono shrink-0"
      onClick={focusInput}
    >
      <span className="text-terminal-prompt select-none">torchleet:~$&nbsp;</span>
      <div className="relative flex-1">
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          className="w-full bg-transparent text-terminal-text outline-none border-none caret-transparent font-mono"
          autoFocus
          spellCheck={false}
          autoComplete="off"
          autoCapitalize="off"
        />
        {/* Blinking block cursor */}
        <span
          className="cursor-blink absolute top-0 text-terminal-prompt pointer-events-none font-mono"
          style={{ left: `${value.length}ch` }}
        >
          &#9608;
        </span>
      </div>
    </div>
  );
}
