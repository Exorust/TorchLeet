import { readFileSync } from "node:fs";
import { join } from "node:path";
import { getRepoRoot } from "./repo-root.js";

export interface ParsedNotebook {
  problemStatement: string;
  functionSignatures: string[];
  todoComments: string[];
  validationCode: string;
}

const cache = new Map<string, ParsedNotebook | null>();

export function parseNotebook(relativePath: string): ParsedNotebook | null {
  if (cache.has(relativePath)) return cache.get(relativePath)!;

  const root = getRepoRoot();
  if (!root) {
    cache.set(relativePath, null);
    return null;
  }

  const fullPath = join(root, relativePath);
  let raw: string;
  try {
    raw = readFileSync(fullPath, "utf-8");
  } catch {
    cache.set(relativePath, null);
    return null;
  }

  let nb: { cells: Array<{ cell_type: string; source: string[] }> };
  try {
    nb = JSON.parse(raw);
  } catch {
    cache.set(relativePath, null);
    return null;
  }

  const mdCells = nb.cells.filter((c) => c.cell_type === "markdown");
  const codeCells = nb.cells.filter((c) => c.cell_type === "code");

  const problemStatement = mdCells
    .map((c) => c.source.join(""))
    .join("\n\n");

  const functionSignatures: string[] = [];
  const todoComments: string[] = [];

  for (const cell of codeCells) {
    for (const line of cell.source) {
      const trimmed = line.trim();
      if (trimmed.startsWith("def ") || trimmed.startsWith("class ")) {
        functionSignatures.push(trimmed);
      }
      if (trimmed.includes("TODO")) {
        todoComments.push(trimmed.replace(/^#\s*/, ""));
      }
    }
  }

  const lastCodeCell = codeCells[codeCells.length - 1];
  const validationCode = lastCodeCell
    ? lastCodeCell.source.join("")
    : "";

  const result: ParsedNotebook = {
    problemStatement,
    functionSignatures,
    todoComments,
    validationCode,
  };

  cache.set(relativePath, result);
  return result;
}

export function getDocstrings(relativePath: string): string[] {
  const root = getRepoRoot();
  if (!root) return [];

  const fullPath = join(root, relativePath);
  let raw: string;
  try {
    raw = readFileSync(fullPath, "utf-8");
  } catch {
    return [];
  }

  let nb: { cells: Array<{ cell_type: string; source: string[] }> };
  try {
    nb = JSON.parse(raw);
  } catch {
    return [];
  }

  const docstrings: string[] = [];
  for (const cell of nb.cells.filter((c) => c.cell_type === "code")) {
    const src = cell.source.join("");
    const matches = src.match(/"""[\s\S]*?"""/g);
    if (matches) docstrings.push(...matches);
  }
  return docstrings;
}
