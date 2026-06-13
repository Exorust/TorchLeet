import { z } from "zod";
import {
  getQuestionsByCompany,
  getAllCompanies,
  getColabUrl,
  type Question,
} from "../data/questions.js";

export const getCompanyPrepSchema = {
  company: z.string().describe("Company name (e.g. 'Anthropic', 'Meta', 'Google')"),
};

const DIFFICULTY_ORDER: Record<string, number> = {
  basic: 0,
  easy: 1,
  medium: 2,
  hard: 3,
  expert: 4,
};

export async function getCompanyPrep(params: { company: string }) {
  const qs = getQuestionsByCompany(params.company);

  if (qs.length === 0) {
    const companies = getAllCompanies();
    return {
      content: [
        {
          type: "text" as const,
          text: `No questions found for "${params.company}".\n\nAvailable companies: ${companies.join(", ")}`,
        },
      ],
    };
  }

  const sorted = [...qs].sort(
    (a, b) => (DIFFICULTY_ORDER[a.difficulty] ?? 5) - (DIFFICULTY_ORDER[b.difficulty] ?? 5),
  );

  const byCategory = new Map<string, Question[]>();
  for (const q of sorted) {
    const cat = q.category || q.set;
    if (!byCategory.has(cat)) byCategory.set(cat, []);
    byCategory.get(cat)!.push(q);
  }

  const parts: string[] = [
    `# ${params.company} Interview Prep\n`,
    `${qs.length} TorchLeet questions tagged for ${params.company}.\n`,
    `Recommended approach: start with Easy/Medium, then work up to Hard/Expert.\n`,
  ];

  parts.push("## Study Plan (by difficulty)\n");
  for (const q of sorted) {
    const colab = q.hasNotebook && q.questionPath ? ` | [Open in Colab](${getColabUrl(q.questionPath)})` : "";
    parts.push(`- **${q.id}** ${q.title} — ${q.difficulty}${colab}`);
  }

  parts.push("\n## By Category\n");
  for (const [cat, catQs] of byCategory) {
    parts.push(`### ${cat}\n`);
    for (const q of catQs) {
      parts.push(`- ${q.id} ${q.title} (${q.difficulty})`);
    }
    parts.push("");
  }

  return { content: [{ type: "text" as const, text: parts.join("\n") }] };
}
