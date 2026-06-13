import { z } from "zod";
import { getQuestionById } from "../data/questions.js";
import { parseNotebook, getDocstrings } from "../lib/notebook-parser.js";

export const getHintSchema = {
  id: z.string().describe("Question ID (e.g. 'v3-1')"),
  level: z
    .number()
    .int()
    .min(1)
    .max(3)
    .default(1)
    .describe("Hint level: 1=conceptual, 2=function signatures, 3=step-by-step TODOs"),
};

export async function getHint(params: { id: string; level: number }) {
  const q = getQuestionById(params.id);
  if (!q) {
    return {
      content: [{ type: "text" as const, text: `Question "${params.id}" not found.` }],
    };
  }

  const level = params.level ?? 1;

  if (q.hasNotebook && q.questionPath) {
    const nb = parseNotebook(q.questionPath);
    if (nb) {
      return notebookHint(q.title, nb, level, q.questionPath);
    }
  }

  return descriptionHint(q.title, q.description, level);
}

function notebookHint(
  title: string,
  nb: ReturnType<typeof parseNotebook> & {},
  level: number,
  questionPath: string,
) {
  const parts: string[] = [`## Hint for: ${title} (Level ${level}/3)\n`];

  if (level >= 1) {
    const statement = nb.problemStatement;
    const truncated = statement.length > 1500
      ? statement.slice(0, 1500) + "\n\n...(problem statement truncated)"
      : statement;
    parts.push("### Problem Statement & Requirements\n");
    parts.push(truncated);
  }

  if (level >= 2) {
    parts.push("\n### Function Signatures & Docstrings\n");
    if (nb.functionSignatures.length > 0) {
      parts.push("You need to implement:\n");
      for (const sig of nb.functionSignatures) {
        parts.push(`\`\`\`python\n${sig}\n\`\`\``);
      }
    }
    const docs = getDocstrings(questionPath);
    if (docs.length > 0) {
      parts.push("\nDocstrings (expected behavior):\n");
      for (const doc of docs) {
        parts.push(`\`\`\`\n${doc}\n\`\`\``);
      }
    }
  }

  if (level >= 3) {
    parts.push("\n### Step-by-Step TODO Guide\n");
    if (nb.todoComments.length > 0) {
      parts.push("Follow these steps:\n");
      for (const todo of nb.todoComments) {
        parts.push(`- ${todo}`);
      }
    } else {
      parts.push("No explicit TODOs in this notebook — review the function signatures and tests carefully.");
    }
  }

  if (level < 3) {
    parts.push(`\n---\n*Need more help? Ask for hint level ${level + 1}.*`);
  }

  return { content: [{ type: "text" as const, text: parts.join("\n") }] };
}

function descriptionHint(title: string, description: string, level: number) {
  const parts: string[] = [`## Hint for: ${title} (Level ${level}/3)\n`];

  if (level >= 1) {
    parts.push("### What to build\n");
    parts.push(description || "No description available for this question.");
  }

  if (level >= 2) {
    parts.push("\n### Approach\n");
    parts.push("Think about:");
    parts.push("- What are the input and output tensor shapes?");
    parts.push("- What PyTorch operations will you need?");
    parts.push("- Are there numerical stability concerns?");
  }

  if (level >= 3) {
    parts.push("\n### Key considerations\n");
    parts.push("- Start by writing the function signature with type hints");
    parts.push("- Test with small tensors first (2x3 or 3x4)");
    parts.push("- Compare your output against PyTorch's built-in implementation");
    parts.push("- Check edge cases: empty tensors, single elements, very large values");
  }

  if (level < 3) {
    parts.push(`\n---\n*Need more help? Ask for hint level ${level + 1}.*`);
  }

  return { content: [{ type: "text" as const, text: parts.join("\n") }] };
}
