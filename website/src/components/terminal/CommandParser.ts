import {
  getQuestionsBySet,
  getQuestionsByCompany,
  getQuestionsByDifficulty,
  getQuestionById,
  getAllCompanies,
  type Question,
  type Difficulty,
} from "@/data/questions";
import { HELP_TEXT } from "@/lib/constants";

export interface CommandResult {
  lines: string[];
  action?: "clear" | "exit" | "open";
  openQuestion?: Question;
}

function pad(str: string, len: number): string {
  return str.length >= len ? str : str + " ".repeat(len - str.length);
}

function formatQuestionTable(questions: Question[]): string[] {
  if (questions.length === 0) {
    return ["  No questions found."];
  }

  const lines: string[] = [];
  lines.push("");
  lines.push(
    `  ${pad("#", 8)}${pad("Title", 52)}${pad("Difficulty", 12)}${questions[0].companies?.length ? "Companies" : ""}`,
  );
  lines.push(`  ${"-".repeat(80)}`);

  for (const q of questions) {
    const companies =
      q.companies.length > 0 ? q.companies.join(", ") : "";
    lines.push(
      `  ${pad(q.id, 8)}${pad(q.title, 52)}${pad(q.difficulty, 12)}${companies}`,
    );
  }

  lines.push("");
  lines.push(`  ${questions.length} question(s)`);
  return lines;
}

export function parseCommand(input: string): CommandResult {
  const trimmed = input.trim();
  if (!trimmed) return { lines: [] };

  const parts = trimmed.split(/\s+/);
  const cmd = parts[0].toLowerCase();
  const arg = parts.slice(1).join(" ");

  switch (cmd) {
    case "help": {
      return { lines: HELP_TEXT.split("\n") };
    }

    case "list":
    case "ls": {
      if (!arg) {
        const v1 = getQuestionsBySet("v1");
        const v2 = getQuestionsBySet("v2");
        const v3 = getQuestionsBySet("v3");
        return {
          lines: [
            "",
            `  [v1] Question Set        — ${v1.length} questions`,
            `  [v2] LLM Set             — ${v2.length} questions`,
            `  [v3] Advanced ML Systems — ${v3.length} questions (NEW!)`,
            "",
            "  Type 'list v1', 'list v2', or 'list v3' to see questions.",
          ],
        };
      }

      const setArg = arg.toLowerCase();
      if (setArg === "v1" || setArg === "v2" || setArg === "v3") {
        const questions = getQuestionsBySet(setArg);
        return { lines: formatQuestionTable(questions) };
      }

      return {
        lines: [`  Unknown set: ${arg}. Available sets: v1, v2, v3`],
      };
    }

    case "open": {
      if (!arg) {
        return {
          lines: ["  Usage: open <id>  (e.g. open v3-14)"],
        };
      }
      const question = getQuestionById(arg.toLowerCase());
      if (!question) {
        return {
          lines: [
            `  Question not found: ${arg}`,
            "  Use 'list' to see available question IDs.",
          ],
        };
      }
      return {
        lines: [`  Opening ${question.id}: ${question.title}...`],
        action: "open",
        openQuestion: question,
      };
    }

    case "company": {
      if (!arg) {
        const companies = getAllCompanies();
        return {
          lines: [
            "",
            "  Available companies:",
            "",
            `  ${companies.join(", ")}`,
            "",
            "  Usage: company <name>  (e.g. company google)",
          ],
        };
      }
      const matches = getQuestionsByCompany(arg);
      if (matches.length === 0) {
        return {
          lines: [
            `  No questions found for company: ${arg}`,
            "  Type 'company' to see all available companies.",
          ],
        };
      }
      return {
        lines: [
          `  Questions tagged with "${arg}":`,
          ...formatQuestionTable(matches),
        ],
      };
    }

    case "filter": {
      if (!arg) {
        return {
          lines: [
            "  Usage: filter <difficulty>",
            "  Levels: basic, easy, medium, hard, expert",
          ],
        };
      }
      const level = arg.toLowerCase() as Difficulty;
      const valid = ["basic", "easy", "medium", "hard", "expert"];
      if (!valid.includes(level)) {
        return {
          lines: [
            `  Unknown difficulty: ${arg}`,
            `  Valid levels: ${valid.join(", ")}`,
          ],
        };
      }
      const matches = getQuestionsByDifficulty(level);
      if (matches.length === 0) {
        return { lines: [`  No questions found with difficulty: ${level}`] };
      }
      return {
        lines: [
          `  ${level.toUpperCase()} questions:`,
          ...formatQuestionTable(matches),
        ],
      };
    }

    case "clear": {
      return { lines: [], action: "clear" };
    }

    case "exit": {
      return { lines: ["  Switching to web mode..."], action: "exit" };
    }

    default: {
      return {
        lines: [
          `  Command not found: ${cmd}. Type 'help' for available commands.`,
        ],
      };
    }
  }
}
