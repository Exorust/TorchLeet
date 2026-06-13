import { z } from "zod";
import { getQuestionById } from "../data/questions.js";
import { parseNotebook } from "../lib/notebook-parser.js";

export const checkSolutionSchema = {
  id: z.string().describe("Question ID"),
  code: z.string().describe("The user's PyTorch solution code"),
};

export async function checkSolution(params: { id: string; code: string }) {
  const q = getQuestionById(params.id);
  if (!q) {
    return {
      content: [{ type: "text" as const, text: `Question "${params.id}" not found.` }],
    };
  }

  const parts: string[] = [
    `## Solution Review Checklist: ${q.title}\n`,
    `**Difficulty**: ${q.difficulty}\n`,
  ];

  if (q.hasNotebook && q.questionPath) {
    const nb = parseNotebook(q.questionPath);
    if (nb) {
      if (nb.functionSignatures.length > 0) {
        parts.push("### Expected Function Signatures\n");
        parts.push("Verify your implementation matches these signatures:\n");
        for (const sig of nb.functionSignatures) {
          parts.push(`- \`${sig}\``);
        }
        parts.push("");
      }

      if (nb.todoComments.length > 0) {
        parts.push("### Required Steps\n");
        parts.push("Make sure you've addressed each TODO:\n");
        for (const todo of nb.todoComments) {
          parts.push(`- [ ] ${todo}`);
        }
        parts.push("");
      }

      if (nb.validationCode) {
        parts.push("### Validation Tests\n");
        parts.push("Run these assertions from the notebook to verify correctness:\n");
        parts.push("```python");
        parts.push(nb.validationCode.slice(0, 2000));
        parts.push("```\n");
      }
    }
  }

  parts.push("### General Checks\n");
  parts.push("- [ ] Tensor shapes are correct at each step");
  parts.push("- [ ] No unnecessary `.detach()` or `.clone()` calls");
  parts.push("- [ ] Gradients flow correctly (no accidental in-place ops)");
  parts.push("- [ ] Numerically stable (no overflow/underflow with large values)");
  parts.push("- [ ] Device-agnostic (would work on both CPU and CUDA)");

  if (q.difficulty === "hard" || q.difficulty === "expert") {
    parts.push("- [ ] Memory efficient (no redundant tensor allocations)");
    parts.push("- [ ] Handles edge cases (empty inputs, batch size 1, etc.)");
  }

  parts.push("\n### Your Code\n");
  parts.push("```python");
  parts.push(params.code);
  parts.push("```");

  return { content: [{ type: "text" as const, text: parts.join("\n") }] };
}
