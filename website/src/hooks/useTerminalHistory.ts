"use client";

import { useState, useCallback, useRef } from "react";

const MAX_HISTORY = 50;

export function useTerminalHistory() {
  const [history, setHistory] = useState<string[]>([]);
  const indexRef = useRef(-1);
  const savedInputRef = useRef("");

  const addToHistory = useCallback((cmd: string) => {
    const trimmed = cmd.trim();
    if (!trimmed) return;
    setHistory((prev) => {
      const next = [...prev, trimmed];
      if (next.length > MAX_HISTORY) next.shift();
      return next;
    });
    indexRef.current = -1;
    savedInputRef.current = "";
  }, []);

  const navigateHistory = useCallback(
    (direction: "up" | "down", currentInput: string): string => {
      if (history.length === 0) return currentInput;

      if (direction === "up") {
        if (indexRef.current === -1) {
          // Save what user was typing before navigating
          savedInputRef.current = currentInput;
          indexRef.current = history.length - 1;
        } else if (indexRef.current > 0) {
          indexRef.current--;
        }
        return history[indexRef.current] ?? currentInput;
      }

      // direction === "down"
      if (indexRef.current === -1) return currentInput;

      if (indexRef.current < history.length - 1) {
        indexRef.current++;
        return history[indexRef.current] ?? currentInput;
      }

      // At the end — restore saved input
      indexRef.current = -1;
      return savedInputRef.current;
    },
    [history],
  );

  return { history, addToHistory, navigateHistory };
}
