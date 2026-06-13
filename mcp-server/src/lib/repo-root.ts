import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

let cached: string | null = null;
let searched = false;

export function getRepoRoot(): string | null {
  if (searched) return cached;
  searched = true;

  if (process.env.TORCHLEET_ROOT) {
    const root = resolve(process.env.TORCHLEET_ROOT);
    if (looksLikeRepo(root)) {
      cached = root;
      return cached;
    }
  }

  const __filename = fileURLToPath(import.meta.url);
  let dir = dirname(__filename);
  for (let i = 0; i < 10; i++) {
    if (looksLikeRepo(dir)) {
      cached = dir;
      return cached;
    }
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }

  return cached;
}

function looksLikeRepo(dir: string): boolean {
  return existsSync(join(dir, "torch")) && existsSync(join(dir, "v3"));
}
