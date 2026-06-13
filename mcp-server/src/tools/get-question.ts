import { z } from "zod";
import { getQuestionById, getColabUrl } from "../data/questions.js";

export const getQuestionSchema = {
  id: z.string().describe("Question ID (e.g. 'v3-1', 'v2-6', 'v1-26')"),
};

export async function getQuestion(params: { id: string }) {
  const q = getQuestionById(params.id);
  if (!q) {
    return {
      content: [{ type: "text" as const, text: `Question "${params.id}" not found. Use list_questions to see available IDs.` }],
    };
  }

  const parts: string[] = [
    `# ${q.title}`,
    `**ID**: ${q.id}`,
    `**Set**: ${q.set} | **Difficulty**: ${q.difficulty}`,
  ];

  if (q.companies.length > 0) {
    parts.push(`**Companies**: ${q.companies.join(", ")}`);
  }
  if (q.category) {
    parts.push(`**Category**: ${q.category}`);
  }
  parts.push(`**Tracks**: ${q.tracks.join(", ")}`);

  if (q.description) {
    parts.push(`\n## Description\n${q.description}`);
  }

  if (q.llmPathOrder) {
    parts.push(`\n**LLM Path**: Step ${q.llmPathOrder} (${q.llmPathStage})`);
  }

  if (q.hasNotebook && q.questionPath) {
    parts.push(`\n## Links`);
    parts.push(`- **Colab (Question)**: ${getColabUrl(q.questionPath)}`);
    if (q.solutionPath) {
      parts.push(`- **Colab (Solution)**: ${getColabUrl(q.solutionPath)}`);
    }
  } else {
    parts.push(`\n*This question does not have a notebook yet.*`);
  }

  return {
    content: [{ type: "text" as const, text: parts.join("\n") }],
  };
}
