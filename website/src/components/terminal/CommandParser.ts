import {
  getQuestionsBySet,
  getQuestionsByCompany,
  getQuestionsByDifficulty,
  getQuestionById,
  getAllCompanies,
  type Question,
  type Difficulty,
} from "@/data/questions";
import {
  isDone,
  markDone,
  markUndone,
  loadProgress,
  getAllQuestions,
  formatProgressLines,
} from "@/lib/progress";
import {
  HELP_TEXT,
  ABOUT_ME,
  GHOST_QUESTION,
  SUDO_DENIED,
  SUDO_ROOT,
  MATRIX_LINES,
  CAT_SECRETS,
  LS_HOME,
} from "@/lib/constants";

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
    `  ${pad("Done", 6)}${pad("#", 8)}${pad("Title", 46)}${pad("Difficulty", 12)}${questions[0].companies?.length ? "Companies" : ""}`,
  );
  lines.push(`  ${"-".repeat(80)}`);

  for (const q of questions) {
    const companies =
      q.companies.length > 0 ? q.companies.join(", ") : "";
    const status = isDone(q.id) ? "[x]" : "[ ]";
    lines.push(
      `  ${pad(status, 6)}${pad(q.id, 8)}${pad(q.title, 46)}${pad(q.difficulty, 12)}${companies}`,
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

    case "list": {
      if (!arg) {
        const v1 = getQuestionsBySet("v1");
        const v2 = getQuestionsBySet("v2");
        const v3 = getQuestionsBySet("v3");
        return {
          lines: [
            "",
            `  [v1] Question Set        - ${v1.length} questions`,
            `  [v2] LLM Set             - ${v2.length} questions`,
            `  [v3] Advanced ML Systems - ${v3.length} questions (NEW!)`,
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
      if (arg.toLowerCase() === "v3-31") {
        return { lines: GHOST_QUESTION.split("\n") };
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

    case "done": {
      if (!arg) {
        return { lines: ["  Usage: done <id>  (e.g. done v3-14)"] };
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
      if (isDone(question.id)) {
        return {
          lines: [
            `  ${question.id} is already marked complete.`,
            `  Use 'undone ${question.id}' to reset.`,
          ],
        };
      }
      markDone(question.id);
      return {
        lines: [
          `  Marked complete: ${question.id} — ${question.title}`,
          "  Nice work! Type 'progress' to see your stats.",
        ],
      };
    }

    case "undone": {
      if (!arg) {
        return { lines: ["  Usage: undone <id>  (e.g. undone v3-14)"] };
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
      if (!isDone(question.id)) {
        return {
          lines: [`  ${question.id} is not marked complete.`],
        };
      }
      markUndone(question.id);
      return {
        lines: [`  Marked incomplete: ${question.id} — ${question.title}`],
      };
    }

    case "status": {
      if (!arg) {
        return { lines: ["  Usage: status <id>  (e.g. status v3-14)"] };
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
      const entry = loadProgress().completed[question.id];
      if (!entry) {
        return {
          lines: [
            `  ${question.id}: ${question.title}`,
            "  Status: [ ] Incomplete",
            `  Use 'done ${question.id}' when you've solved it.`,
          ],
        };
      }
      const date = new Date(entry.completedAt).toLocaleDateString();
      return {
        lines: [
          `  ${question.id}: ${question.title}`,
          `  Status: [x] Complete (completed ${date})`,
        ],
      };
    }

    case "progress": {
      return { lines: formatProgressLines(getAllQuestions()) };
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

    case "aboutme": {
      return { lines: ABOUT_ME.split("\n") };
    }

    case "feedback": {
      if (!arg) {
        return {
          lines: [
            "  Usage: feedback <your message>",
            "  Your feedback will be sent to the creator.",
          ],
        };
      }
      const subject = encodeURIComponent("TorchLeet Feedback");
      const body = encodeURIComponent(arg);
      if (typeof window !== "undefined") {
        window.open(
          `mailto:chandrahas.aroori@gmail.com?subject=${subject}&body=${body}`,
          "_blank",
        );
      }
      return {
        lines: [
          "  Opening your email client...",
          "  Thanks for the feedback!",
        ],
      };
    }

    case "sudo": {
      if (arg.toLowerCase() === "torchleet" || arg.toLowerCase() === "-i torchleet") {
        return { lines: SUDO_ROOT.split("\n") };
      }
      return { lines: SUDO_DENIED.split("\n") };
    }

    case "matrix": {
      return { lines: MATRIX_LINES };
    }

    case "cat": {
      if (arg.includes(".secret") || arg.includes("secret")) {
        return { lines: CAT_SECRETS.split("\n") };
      }
      return {
        lines: [`  cat: ${arg || "(no file)"}: Permission denied`],
      };
    }

    case "ls": {
      if (arg.includes("/home") || arg.includes("~") || arg === "-la" || arg === "-a") {
        return { lines: LS_HOME.split("\n") };
      }
      if (!arg) {
        const v1 = getQuestionsBySet("v1");
        const v2 = getQuestionsBySet("v2");
        const v3 = getQuestionsBySet("v3");
        return {
          lines: [
            "",
            `  [v1] Question Set        - ${v1.length} questions`,
            `  [v2] LLM Set             - ${v2.length} questions`,
            `  [v3] Advanced ML Systems - ${v3.length} questions (NEW!)`,
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

    case "github": {
      if (typeof window !== "undefined") {
        window.open("https://github.com/Exorust/TorchLeet", "_blank");
      }
      return {
        lines: [
          "  Opening github.com/Exorust/TorchLeet...",
          "  Star the repo if you find it useful!",
        ],
      };
    }

    case "pwd": {
      return { lines: ["  /home/torch/torchleet"] };
    }

    case "cd": {
      return { lines: ["  Nice try. You're staying right here."] };
    }

    case "rm": {
      return { lines: ["  Permission denied. No deleting questions."] };
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
