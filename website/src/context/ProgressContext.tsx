"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { type Question } from "@/data/questions";
import {
  getAllQuestions,
  loadProgress,
  markDone as markDoneStore,
  markUndone as markUndoneStore,
  getProgressStats,
  PROGRESS_UPDATE_EVENT,
  type ProgressEntry,
  type ProgressStats,
} from "@/lib/progress";

interface ProgressState {
  completed: Record<string, ProgressEntry>;
  isDone: (id: string) => boolean;
  markDone: (id: string) => boolean;
  markUndone: (id: string) => boolean;
  getStats: (questions?: Question[]) => ProgressStats;
}

const ProgressContext = createContext<ProgressState | null>(null);

export function ProgressProvider({ children }: { children: ReactNode }) {
  const [completed, setCompleted] = useState<Record<string, ProgressEntry>>(
    {},
  );

  const refresh = useCallback(() => {
    setCompleted(loadProgress().completed);
  }, []);

  useEffect(() => {
    refresh();
    const onUpdate = () => refresh();
    window.addEventListener(PROGRESS_UPDATE_EVENT, onUpdate);
    window.addEventListener("storage", onUpdate);
    return () => {
      window.removeEventListener(PROGRESS_UPDATE_EVENT, onUpdate);
      window.removeEventListener("storage", onUpdate);
    };
  }, [refresh]);

  const isDone = useCallback(
    (id: string) => id in completed,
    [completed],
  );

  const markDone = useCallback(
    (id: string) => {
      const changed = markDoneStore(id);
      if (changed) refresh();
      return changed;
    },
    [refresh],
  );

  const markUndone = useCallback(
    (id: string) => {
      const changed = markUndoneStore(id);
      if (changed) refresh();
      return changed;
    },
    [refresh],
  );

  const getStats = useCallback(
    (questions?: Question[]) => getProgressStats(questions ?? getAllQuestions()),
    [],
  );

  const value = useMemo(
    () => ({ completed, isDone, markDone, markUndone, getStats }),
    [completed, isDone, markDone, markUndone, getStats],
  );

  return (
    <ProgressContext.Provider value={value}>
      {children}
    </ProgressContext.Provider>
  );
}

export function useProgress() {
  const ctx = useContext(ProgressContext);
  if (!ctx) {
    throw new Error("useProgress must be used within ProgressProvider");
  }
  return ctx;
}
