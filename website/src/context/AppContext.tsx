"use client";

import { createContext, useContext, useState, ReactNode } from "react";

type QuestionSet = "v1" | "v2" | "v3";
type V3Category =
  | "classical-ml"
  | "llm-decoding"
  | "llm-inference"
  | "modern-architectures"
  | "alignment-training"
  | "gpu-systems";

interface AppState {
  mode: "terminal" | "web";
  setMode: (mode: "terminal" | "web") => void;
  activeSet: QuestionSet;
  setActiveSet: (s: QuestionSet) => void;
  activeV3Category: V3Category | null;
  setActiveV3Category: (c: V3Category | null) => void;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<"terminal" | "web">("web");
  const [activeSet, setActiveSet] = useState<QuestionSet>("v3");
  const [activeV3Category, setActiveV3Category] =
    useState<V3Category | null>(null);

  return (
    <AppContext.Provider
      value={{
        mode,
        setMode,
        activeSet,
        setActiveSet,
        activeV3Category,
        setActiveV3Category,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within AppProvider");
  return ctx;
}
