import { z } from "zod";
import {
  getLLMPathStages,
  getBasicsQuestions,
  getAdvancedGroupedByTopic,
} from "../data/questions.js";

export const getLearningPathSchema = {
  track: z
    .enum(["llm-path", "basics", "advanced"])
    .default("llm-path")
    .describe("Which learning track to show"),
};

export async function getLearningPath(params: { track: string }) {
  if (params.track === "llm-path") {
    const stages = getLLMPathStages();
    const parts = [
      "# LLM Learning Path — Build a Language Model from Scratch\n",
      "Work through these in order. Each builds on the previous.\n",
    ];

    for (const stage of stages) {
      parts.push(`## ${stage.title}\n`);
      for (const q of stage.questions) {
        const check = q.hasNotebook ? "[ ]" : "[no notebook]";
        parts.push(`- ${check} **${q.id}** ${q.title} (${q.difficulty})`);
      }
      parts.push("");
    }

    return { content: [{ type: "text" as const, text: parts.join("\n") }] };
  }

  if (params.track === "basics") {
    const qs = getBasicsQuestions();
    const parts = [
      `# Basics Track — ${qs.length} questions\n`,
      "Core PyTorch and classical ML fundamentals.\n",
    ];
    for (const q of qs) {
      parts.push(`- **${q.id}** ${q.title} (${q.difficulty})`);
    }
    return { content: [{ type: "text" as const, text: parts.join("\n") }] };
  }

  // advanced
  const groups = getAdvancedGroupedByTopic();
  const parts = [
    "# Advanced Track — Grouped by Topic\n",
    "Hard and expert-level problems for deep practice.\n",
  ];
  for (const group of groups) {
    parts.push(`## ${group.title}\n`);
    for (const q of group.questions) {
      const companies = q.companies.length > 0 ? ` [${q.companies.join(", ")}]` : "";
      parts.push(`- **${q.id}** ${q.title} (${q.difficulty})${companies}`);
    }
    parts.push("");
  }

  return { content: [{ type: "text" as const, text: parts.join("\n") }] };
}
