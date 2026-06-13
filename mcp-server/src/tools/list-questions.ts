import { z } from "zod";
import {
  questions,
  type Question,
  type QuestionSet,
  type Difficulty,
} from "../data/questions.js";

export const listQuestionsSchema = {
  set: z.enum(["v1", "v2", "v3"]).optional().describe("Filter by question set"),
  difficulty: z
    .enum(["basic", "easy", "medium", "hard", "expert"])
    .optional()
    .describe("Filter by difficulty level"),
  company: z.string().optional().describe("Filter by company name (e.g. 'Anthropic', 'Meta')"),
  category: z.string().optional().describe("Filter by v3 category (e.g. 'gpu-systems', 'llm-decoding')"),
  track: z
    .enum(["basics", "advanced", "llm-path"])
    .optional()
    .describe("Filter by learning track"),
};

export async function listQuestions(params: {
  set?: string;
  difficulty?: string;
  company?: string;
  category?: string;
  track?: string;
}) {
  let filtered: Question[] = [...questions];

  if (params.set) {
    filtered = filtered.filter((q) => q.set === params.set as QuestionSet);
  }
  if (params.difficulty) {
    filtered = filtered.filter((q) => q.difficulty === params.difficulty as Difficulty);
  }
  if (params.company) {
    const lower = params.company.toLowerCase();
    filtered = filtered.filter((q) =>
      q.companies.some((c) => c.toLowerCase().includes(lower)),
    );
  }
  if (params.category) {
    filtered = filtered.filter((q) => q.category === params.category);
  }
  if (params.track) {
    const track = params.track as "basics" | "advanced" | "llm-path";
    filtered = filtered.filter((q) => q.tracks.includes(track));
  }

  const lines = filtered.map(
    (q) =>
      `${q.id} | ${q.title} | ${q.difficulty} | ${q.companies.length > 0 ? q.companies.join(", ") : "—"} | ${q.hasNotebook ? "📓" : "—"}`,
  );

  const header = `Found ${filtered.length} question(s)\n\nID | Title | Difficulty | Companies | Notebook\n---|-------|------------|-----------|--------`;

  return {
    content: [{ type: "text" as const, text: `${header}\n${lines.join("\n")}` }],
  };
}
