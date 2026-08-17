// Company-wise views over the generated problem data.
// Source of truth: per-problem problem.toml manifests -> problems.generated.ts.
// Pure helpers only; never edit problems.generated.ts by hand.

import {
  generatedProblems,
  type GeneratedProblem,
} from "@/data/problems.generated";

export interface CompanySummary {
  name: string;
  slug: string;
  count: number;
  byDifficulty: Record<string, number>;
  hasReported: boolean;
  /** Legacy question ids ("v3-1", …) used by the progress store. */
  problemIds: string[];
}

export interface CompanyDetail {
  name: string;
  slug: string;
  problems: GeneratedProblem[];
  hasReported: boolean;
}

const DIFFICULTY_ORDER = ["basic", "easy", "medium", "hard", "expert"];

/**
 * A company needs this many tagged problems before it gets its own page.
 *
 * A page listing one or two problems is a thin doorway page: it competes with
 * our own problem pages, gives a visitor nothing the index does not, and is the
 * pattern search engines discount. Companies below the bar still appear as tags
 * and on the company index, they just do not get a route. Lower this as the
 * interview-report backlog grows.
 */
export const COMPANY_PAGE_MIN_PROBLEMS = 5;

export function companySlug(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export function getCompanies(): CompanySummary[] {
  const byCompany = new Map<string, CompanySummary>();

  for (const p of generatedProblems) {
    for (const name of p.companies) {
      let summary = byCompany.get(name);
      if (!summary) {
        summary = {
          name,
          slug: companySlug(name),
          count: 0,
          byDifficulty: {},
          hasReported: false,
          problemIds: [],
        };
        byCompany.set(name, summary);
      }
      summary.count += 1;
      summary.problemIds.push(...p.legacyIds);
      summary.byDifficulty[p.difficulty] =
        (summary.byDifficulty[p.difficulty] ?? 0) + 1;
      if (p.companyConfidence[name] === "reported") summary.hasReported = true;
    }
  }

  return Array.from(byCompany.values()).sort(
    (a, b) => b.count - a.count || a.name.localeCompare(b.name),
  );
}

/** Companies substantial enough to warrant their own page. */
export function getCompaniesWithPages(): CompanySummary[] {
  return getCompanies().filter((c) => c.count >= COMPANY_PAGE_MIN_PROBLEMS);
}

/** Does this company have a page? Use before linking, or the link 404s. */
export function hasCompanyPage(name: string): boolean {
  const c = getCompanies().find((x) => x.name === name);
  return !!c && c.count >= COMPANY_PAGE_MIN_PROBLEMS;
}

export function getCompanyBySlug(slug: string): CompanyDetail | undefined {
  const summary = getCompaniesWithPages().find((c) => c.slug === slug);
  if (!summary) return undefined;

  const problems = generatedProblems
    .filter((p) => p.companies.includes(summary.name))
    .sort(
      (a, b) =>
        DIFFICULTY_ORDER.indexOf(a.difficulty) -
          DIFFICULTY_ORDER.indexOf(b.difficulty) || a.title.localeCompare(b.title),
    );

  return {
    name: summary.name,
    slug: summary.slug,
    problems,
    hasReported: summary.hasReported,
  };
}

export function difficultyBreakdown(
  byDifficulty: Record<string, number>,
): string {
  return DIFFICULTY_ORDER.filter((d) => byDifficulty[d])
    .map((d) => `${byDifficulty[d]} ${d}`)
    .join(" · ");
}
