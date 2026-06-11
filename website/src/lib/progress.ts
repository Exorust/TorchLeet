import {
  getQuestionsBySet,
  type Question,
  type Difficulty,
} from "@/data/questions";

const STORAGE_KEY = "torchleet-progress";
export const PROGRESS_UPDATE_EVENT = "torchleet-progress-update";

export interface ProgressEntry {
  completedAt: string;
}

export interface ProgressStore {
  completed: Record<string, ProgressEntry>;
}

export type MeterDifficulty = "easy" | "medium" | "hard";

export interface DifficultyStats {
  done: number;
  total: number;
  percent: number;
}

export interface ProgressStats {
  completed: number;
  total: number;
  percent: number;
  byDifficulty: Record<MeterDifficulty, DifficultyStats>;
}

const METER_BUCKETS: Record<Difficulty, MeterDifficulty> = {
  basic: "easy",
  easy: "easy",
  medium: "medium",
  hard: "hard",
  expert: "hard",
};

export function getAllQuestions(): Question[] {
  return [
    ...getQuestionsBySet("v1"),
    ...getQuestionsBySet("v2"),
    ...getQuestionsBySet("v3"),
  ];
}

function emptyStore(): ProgressStore {
  return { completed: {} };
}

export function loadProgress(): ProgressStore {
  if (typeof window === "undefined") return emptyStore();
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return emptyStore();
    const parsed = JSON.parse(raw) as ProgressStore;
    return parsed?.completed ? parsed : emptyStore();
  } catch {
    return emptyStore();
  }
}

export function saveProgress(store: ProgressStore): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  window.dispatchEvent(new CustomEvent(PROGRESS_UPDATE_EVENT));
}

export function isDone(id: string): boolean {
  return id in loadProgress().completed;
}

export function markDone(id: string): boolean {
  const store = loadProgress();
  if (store.completed[id]) return false;
  store.completed[id] = { completedAt: new Date().toISOString() };
  saveProgress(store);
  return true;
}

export function markUndone(id: string): boolean {
  const store = loadProgress();
  if (!store.completed[id]) return false;
  delete store.completed[id];
  saveProgress(store);
  return true;
}

export function getProgressStats(questions: Question[]): ProgressStats {
  const completed = loadProgress().completed;
  const byDifficulty: Record<MeterDifficulty, DifficultyStats> = {
    easy: { done: 0, total: 0, percent: 0 },
    medium: { done: 0, total: 0, percent: 0 },
    hard: { done: 0, total: 0, percent: 0 },
  };

  let doneCount = 0;
  for (const q of questions) {
    const bucket = METER_BUCKETS[q.difficulty];
    byDifficulty[bucket].total += 1;
    if (completed[q.id]) {
      doneCount += 1;
      byDifficulty[bucket].done += 1;
    }
  }

  const total = questions.length;
  for (const bucket of Object.keys(byDifficulty) as MeterDifficulty[]) {
    const stats = byDifficulty[bucket];
    stats.percent =
      stats.total > 0 ? Math.round((stats.done / stats.total) * 100) : 0;
  }

  return {
    completed: doneCount,
    total,
    percent: total > 0 ? Math.round((doneCount / total) * 100) : 0,
    byDifficulty,
  };
}

function formatBar(done: number, total: number, width = 20): string {
  if (total === 0) return "░".repeat(width);
  const filled = Math.round((done / total) * width);
  return "█".repeat(filled) + "░".repeat(width - filled);
}

export function formatProgressLines(questions: Question[]): string[] {
  const stats = getProgressStats(questions);
  const { byDifficulty } = stats;

  return [
    "",
    `  Progress: ${stats.completed} / ${stats.total} (${stats.percent}%)`,
    "",
    `  Easy   [${formatBar(byDifficulty.easy.done, byDifficulty.easy.total)}] ${byDifficulty.easy.done}/${byDifficulty.easy.total} (${byDifficulty.easy.percent}%)`,
    `  Medium [${formatBar(byDifficulty.medium.done, byDifficulty.medium.total)}] ${byDifficulty.medium.done}/${byDifficulty.medium.total} (${byDifficulty.medium.percent}%)`,
    `  Hard   [${formatBar(byDifficulty.hard.done, byDifficulty.hard.total)}] ${byDifficulty.hard.done}/${byDifficulty.hard.total} (${byDifficulty.hard.percent}%)`,
    "",
    "  Use 'done <id>' to mark a question complete.",
    "  Use 'undone <id>' to mark it incomplete.",
  ];
}
