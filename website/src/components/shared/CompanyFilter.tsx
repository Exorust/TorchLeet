"use client";

import { useMemo, useState } from "react";
import { getAllCompanies } from "@/data/questions";
import { useApp } from "@/context/AppContext";

interface CompanyFilterProps {
  className?: string;
  compact?: boolean;
}

// Prominent / "big" companies shown by default, in priority order (not alphabetical).
// These are the labs and companies most relevant for ML/AI interviews.
const BIG_COMPANIES = [
  "Anthropic",
  "OpenAI",
  "Google",
  "Meta",
  "DeepMind",
  "NVIDIA",
  "xAI",
  "Apple",
  "Mistral",
  "Perplexity",
  "Together AI",
  "Databricks",
];

export default function CompanyFilter({ className = "", compact = false }: CompanyFilterProps) {
  const { companyFilters, toggleCompanyFilter, clearCompanyFilters } = useApp();
  const allCompanies = useMemo(() => getAllCompanies(), []);
  const [showAll, setShowAll] = useState(false);

  if (allCompanies.length === 0) return null;

  const selected = companyFilters;
  const displayedCompanies = showAll ? allCompanies : BIG_COMPANIES;

  return (
    <div className={`space-y-2 ${className}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="text-xs font-semibold uppercase tracking-wider text-foreground/50">
          Filter by company {selected.length > 0 && `(${selected.length})`}
        </div>
        {selected.length > 0 && (
          <button
            onClick={clearCompanyFilters}
            className="text-[10px] font-medium text-lavender-600 hover:text-lavender-700 transition"
          >
            Clear all
          </button>
        )}
      </div>

      <div className="flex flex-wrap gap-1.5">
        {displayedCompanies.map((company) => {
          const isActive = selected.includes(company);
          return (
            <button
              key={company}
              onClick={() => toggleCompanyFilter(company)}
              className={`rounded-full px-3 py-1 text-xs font-medium transition-all border ${
                isActive
                  ? "bg-lavender-600 text-white border-lavender-600 shadow-sm"
                  : "bg-white/70 text-foreground/70 border-white/60 hover:bg-white hover:text-foreground hover:border-lavender-200"
              } ${compact ? "px-2.5 py-0.5 text-[10px]" : ""}`}
              aria-pressed={isActive}
            >
              {company}
            </button>
          );
        })}
      </div>

      {!showAll && allCompanies.length > BIG_COMPANIES.length && (
        <button
          onClick={() => setShowAll(true)}
          className="text-[10px] font-medium text-lavender-600 hover:text-lavender-700 underline-offset-2 hover:underline transition"
        >
          Show all companies...
        </button>
      )}

      {showAll && (
        <button
          onClick={() => setShowAll(false)}
          className="text-[10px] font-medium text-foreground/60 hover:text-foreground/80 underline-offset-2 hover:underline transition"
        >
          Show top companies only
        </button>
      )}

      {selected.length > 0 && (
        <p className="text-[10px] text-foreground/50">
          Showing questions asked by selected companies (OR).
        </p>
      )}
    </div>
  );
}
