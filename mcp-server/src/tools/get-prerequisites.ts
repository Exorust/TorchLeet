import { z } from "zod";
import { getQuestionById, questions, type Question } from "../data/questions.js";

export const getPrerequisitesSchema = {
  id: z.string().describe("Question ID to find prerequisites for"),
};

const PREREQ_GRAPH: Record<string, string[]> = {
  // Attention chain
  "v2-6": [],
  "v2-7": ["v2-6"],
  "v2-8": ["v2-7"],
  "v3-13": ["v2-7"],
  "v3-12": ["v2-7"],

  // Positional embeddings
  "v2-10": [],
  "v2-11": ["v2-10"],

  // SmolLM capstone
  "v2-12": ["v2-3", "v2-11", "v2-8", "v3-12"],

  // Alignment chain
  "v3-11": [],
  "v3-14": ["v3-11"],
  "v3-15": ["v3-14"],
  "v3-27": ["v3-15"],

  // Decoding chain
  "v3-10": [],
  "v3-8": ["v3-10"],
  "v3-7": ["v3-10"],
  "v3-9": ["v3-10"],

  // Inference chain
  "v3-18": ["v3-12"],
  "v3-19": ["v3-12"],
  "v3-28": ["v3-18", "v3-19"],

  // GPU systems chain
  "v3-1": [],
  "v3-24": ["v3-1"],
  "v3-25": ["v2-6", "v3-24"],
  "v3-26": [],
  "v3-30": ["v2-7", "v3-26"],

  // Modern architectures
  "v3-5": [],
  "v3-6": ["v2-10"],
  "v3-17": [],
  "v3-20": [],
  "v3-21": ["v3-20"],
  "v3-22": [],
  "v3-23": ["v3-6"],
  "v3-29": [],

  // Diffusion
  "v3-16": [],
};

const DIFFICULTY_ORDER: Record<string, number> = {
  basic: 0,
  easy: 1,
  medium: 2,
  hard: 3,
  expert: 4,
};

export async function getPrerequisites(params: { id: string }) {
  const q = getQuestionById(params.id);
  if (!q) {
    return {
      content: [{ type: "text" as const, text: `Question "${params.id}" not found.` }],
    };
  }

  const prereqIds = PREREQ_GRAPH[params.id];

  if (prereqIds && prereqIds.length > 0) {
    const prereqs = prereqIds
      .map((id) => getQuestionById(id))
      .filter((p): p is Question => p !== undefined);

    const parts = [
      `## Prerequisites for: ${q.title} (${q.difficulty})\n`,
      "Complete these first:\n",
    ];
    for (const p of prereqs) {
      parts.push(`1. **${p.id}** ${p.title} (${p.difficulty}) — ${p.description}`);
    }
    return { content: [{ type: "text" as const, text: parts.join("\n") }] };
  }

  if (prereqIds && prereqIds.length === 0) {
    return {
      content: [
        {
          type: "text" as const,
          text: `## ${q.title}\n\nNo prerequisites — this is a good starting point! Just make sure you're comfortable with basic PyTorch tensor operations.`,
        },
      ],
    };
  }

  // Not in the graph: suggest easier questions from the same category/set
  const sameArea = questions.filter(
    (other) =>
      other.id !== q.id &&
      ((q.category && other.category === q.category) || other.set === q.set) &&
      (DIFFICULTY_ORDER[other.difficulty] ?? 5) < (DIFFICULTY_ORDER[q.difficulty] ?? 5),
  );

  if (sameArea.length === 0) {
    return {
      content: [
        {
          type: "text" as const,
          text: `## ${q.title}\n\nNo specific prerequisites mapped. Start with the basics track if you're new to PyTorch.`,
        },
      ],
    };
  }

  const sorted = sameArea
    .sort((a, b) => (DIFFICULTY_ORDER[a.difficulty] ?? 5) - (DIFFICULTY_ORDER[b.difficulty] ?? 5))
    .slice(0, 5);

  const parts = [
    `## Suggested preparation for: ${q.title} (${q.difficulty})\n`,
    "These easier questions cover related concepts:\n",
  ];
  for (const p of sorted) {
    parts.push(`- **${p.id}** ${p.title} (${p.difficulty})`);
  }

  return { content: [{ type: "text" as const, text: parts.join("\n") }] };
}
