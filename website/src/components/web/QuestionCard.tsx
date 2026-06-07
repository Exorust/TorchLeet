"use client";

import { type Question } from "@/data/questions";
import DifficultyBadge from "@/components/shared/DifficultyBadge";
import CompanyTag from "@/components/shared/CompanyTag";

interface Props {
  question: Question;
  onClick: () => void;
}

export default function QuestionCard({ question, onClick }: Props) {
  const setBadgeColor: Record<string, string> = {
    v1: "bg-gray-100 text-gray-600",
    v2: "bg-blue-50 text-blue-600",
    v3: "bg-lavender-100 text-lavender-700",
  };

  return (
    <div
      onClick={onClick}
      className={`bg-white rounded-xl shadow-sm border border-lavender-100 p-5 hover:shadow-md hover:border-lavender-300 transition-all cursor-pointer ${
        !question.hasNotebook ? "opacity-60" : ""
      }`}
    >
      {/* Top row */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-lavender-400 text-sm font-mono">
          #{question.number}
        </span>
        <div className="flex items-center gap-2">
          <span
            className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase ${setBadgeColor[question.set]}`}
          >
            {question.set}
          </span>
          {!question.hasNotebook && (
            <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-gray-100 text-gray-500">
              Coming Soon
            </span>
          )}
        </div>
      </div>

      {/* Title */}
      <h3 className="text-gray-900 font-medium text-base mb-1 line-clamp-2">
        {question.title}
      </h3>

      {/* Description */}
      {question.description && (
        <p className="text-gray-500 text-sm mb-3 line-clamp-2">
          {question.description}
        </p>
      )}

      {/* Bottom row */}
      <div className="flex items-center justify-between gap-2">
        <DifficultyBadge difficulty={question.difficulty} />
        {question.companies.length > 0 && (
          <div className="flex flex-wrap gap-1 justify-end">
            {question.companies.map((company) => (
              <CompanyTag key={company} company={company} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
