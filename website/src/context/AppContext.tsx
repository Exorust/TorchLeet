"use client";

import { createContext, useContext, useState, ReactNode, useCallback } from "react";

export type ActiveView = "llm-path" | "basics" | "advanced";

// Keep old types for terminal compatibility during transition
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

  // New primary navigation state (revamp)
  activeView: ActiveView;
  setActiveView: (view: ActiveView) => void;

  // Cross-cutting company filter (multi-select)
  companyFilters: string[];
  setCompanyFilters: (companies: string[]) => void;
  toggleCompanyFilter: (company: string) => void;
  clearCompanyFilters: () => void;

  // Legacy (kept for terminal + gradual migration)
  activeSet: QuestionSet;
  setActiveSet: (s: QuestionSet) => void;
  activeV3Category: V3Category | null;
  setActiveV3Category: (c: V3Category | null) => void;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<"terminal" | "web">("web");

  // New revamp state
  const [activeView, setActiveView] = useState<ActiveView>("llm-path");
  const [companyFilters, setCompanyFilters] = useState<string[]>([]);

  const toggleCompanyFilter = useCallback((company: string) => {
    setCompanyFilters((prev) =>
      prev.includes(company)
        ? prev.filter((c) => c !== company)
        : [...prev, company]
    );
  }, []);

  const clearCompanyFilters = useCallback(() => {
    setCompanyFilters([]);
  }, []);

  // Legacy state (still used by terminal commands and some helpers)
  const [activeSet, setActiveSet] = useState<QuestionSet>("v3");
  const [activeV3Category, setActiveV3Category] =
    useState<V3Category | null>(null);

  return (
    <AppContext.Provider
      value={{
        mode,
        setMode,
        activeView,
        setActiveView,
        companyFilters,
        setCompanyFilters,
        toggleCompanyFilter,
        clearCompanyFilters,
        // legacy
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
