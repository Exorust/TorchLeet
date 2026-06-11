"use client";

import type { Question } from "@/data/questions";
import { getColabUrl, getDownloadUrl } from "@/data/questions";
import { useProgress } from "@/context/ProgressContext";
import DifficultyBadge from "@/components/shared/DifficultyBadge";
import CompanyTag from "@/components/shared/CompanyTag";

interface Props {
  question: Question | null;
  onClose: () => void;
}

function ActionButton({
  href,
  children,
}: {
  href: string;
  children: React.ReactNode;
}) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="block w-full rounded border border-lavender-700/40 bg-lavender-900/20 px-4 py-2.5 text-center text-sm text-lavender-300 transition-colors hover:bg-lavender-800/30 hover:text-lavender-200 font-mono"
    >
      {children}
    </a>
  );
}

export default function SidePanel({ question, onClose }: Props) {
  const { isDone, markDone, markUndone } = useProgress();
  const isOpen = question !== null;

  return (
    <>
      {/* Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-40"
          onClick={onClose}
        />
      )}

      {/* Panel */}
      <div
        className={`fixed top-0 right-0 z-50 h-full w-[400px] max-w-full bg-terminal-panel border-l border-lavender-700/40 flex flex-col transition-transform duration-300 ease-in-out ${
          isOpen ? "translate-x-0" : "translate-x-full"
        }`}
      >
        {question && (
          <>
            {/* Header */}
            <div className="flex items-start justify-between p-6 pb-4">
              <div className="flex-1 pr-4">
                <p className="text-terminal-dim text-xs font-mono uppercase tracking-wider mb-2">
                  {question.id}
                </p>
                <h2 className="text-lavender-200 text-lg font-bold font-mono leading-snug">
                  {question.title}
                </h2>
              </div>
              <button
                onClick={onClose}
                className="text-terminal-dim hover:text-terminal-text transition-colors text-xl leading-none p-1"
                aria-label="Close panel"
              >
                &#x2715;
              </button>
            </div>

            {/* Description */}
            {question.description && (
              <div className="px-6 pb-4">
                <p className="text-terminal-text text-sm font-mono leading-relaxed">
                  {question.description}
                </p>
              </div>
            )}

            {/* Meta */}
            <div className="px-6 pb-4 flex flex-wrap items-center gap-2">
              <DifficultyBadge difficulty={question.difficulty} />
              {isDone(question.id) && (
                <span className="text-xs font-mono text-green-400">
                  [x] Complete
                </span>
              )}
              {question.category && (
                <span className="text-terminal-dim text-xs font-mono">
                  {question.category}
                </span>
              )}
            </div>

            {/* Companies */}
            {question.companies.length > 0 && (
              <div className="px-6 pb-4 flex flex-wrap gap-1.5">
                {question.companies.map((c) => (
                  <CompanyTag key={c} company={c} />
                ))}
              </div>
            )}

            {/* Divider */}
            <div className="mx-6 border-t border-lavender-700/30" />

            {/* Actions */}
            <div className="p-6 flex flex-col gap-3">
              {question.hasNotebook && question.questionPath ? (
                <>
                  <ActionButton href={getColabUrl(question.questionPath)}>
                    Open Question in Colab
                  </ActionButton>
                  {question.solutionPath && (
                    <ActionButton href={getColabUrl(question.solutionPath)}>
                      Open Solution in Colab
                    </ActionButton>
                  )}
                  <ActionButton href={getDownloadUrl(question.questionPath)}>
                    Download Question
                  </ActionButton>
                  {question.solutionPath && (
                    <ActionButton
                      href={getDownloadUrl(question.solutionPath)}
                    >
                      Download Solution
                    </ActionButton>
                  )}
                </>
              ) : (
                <div className="rounded border border-terminal-dim/30 bg-terminal-bg px-4 py-6 text-center">
                  <p className="text-terminal-dim text-sm font-mono">
                    Notebook coming soon
                  </p>
                </div>
              )}
              <button
                type="button"
                onClick={() =>
                  isDone(question.id)
                    ? markUndone(question.id)
                    : markDone(question.id)
                }
                className={`w-full rounded border px-4 py-2.5 text-center text-sm font-mono transition-colors ${
                  isDone(question.id)
                    ? "border-lavender-700/40 text-terminal-dim hover:text-terminal-text"
                    : "border-lavender-500/60 bg-lavender-900/30 text-lavender-200 hover:bg-lavender-800/40"
                }`}
              >
                {isDone(question.id)
                  ? "Mark as Incomplete"
                  : "Mark as Complete"}
              </button>
            </div>
          </>
        )}
      </div>
    </>
  );
}
