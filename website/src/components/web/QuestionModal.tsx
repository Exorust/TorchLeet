"use client";

import { useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { type Question, getColabUrl, getDownloadUrl } from "@/data/questions";
import { useProgress } from "@/context/ProgressContext";
import DifficultyBadge from "@/components/shared/DifficultyBadge";
import CompanyTag from "@/components/shared/CompanyTag";

interface Props {
  question: Question;
  onClose: () => void;
}

export default function QuestionModal({ question, onClose }: Props) {
  const { isDone, markDone, markUndone } = useProgress();

  const handleEscape = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    },
    [onClose],
  );

  useEffect(() => {
    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [handleEscape]);

  const done = isDone(question.id);

  return (
    <motion.div
      className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      onClick={onClose}
    >
      <motion.div
        layoutId={`card-${question.id}`}
        className="bg-white/70 backdrop-blur-xl rounded-3xl max-w-lg w-full mx-4 p-6 shadow-xl border border-white/50 relative"
        onClick={(e) => e.stopPropagation()}
      >
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.25, delay: 0.15 }}
        >
          {/* Close button */}
          <button
            onClick={onClose}
            className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 transition text-xl leading-none"
            aria-label="Close"
          >
            &times;
          </button>

          {/* Header */}
          <div className="pr-8 space-y-3 mb-4">
            <h2 className="text-xl font-bold text-gray-900">
              {question.title}
            </h2>

            <div className="flex items-center gap-2 flex-wrap">
              <DifficultyBadge difficulty={question.difficulty} />
              {done && (
                <span className="text-xs font-medium px-2.5 py-0.5 rounded-full bg-green-100 text-green-700">
                  Completed
                </span>
              )}
            </div>

            {question.companies.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {question.companies.map((company) => (
                  <CompanyTag key={company} company={company} />
                ))}
              </div>
            )}
          </div>

          {/* Description */}
          {question.description && (
            <p className="text-gray-600 text-sm leading-relaxed mb-4">
              {question.description}
            </p>
          )}

          {/* Divider */}
          <hr className="border-gray-200 mb-5" />

          {/* Action area styled like the white cards */}
          <div className="bg-white/60 backdrop-blur-lg rounded-2xl border border-white/50 p-4">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-gray-500 mb-2 px-1">
              Primary actions
            </div>

            {/* Colab primary — softer card-like treatment */}
            <div className="grid grid-cols-2 gap-3 mb-3">
              <a
                href={
                  question.questionPath
                    ? getColabUrl(question.questionPath)
                    : undefined
                }
                target="_blank"
                rel="noopener noreferrer"
                className={`flex items-center justify-center gap-2 rounded-2xl px-4 py-3 text-sm font-medium transition shadow-sm ${
                  question.questionPath
                    ? "bg-lavender-600 text-white hover:bg-lavender-700 active:scale-[0.985]"
                    : "bg-gray-100 text-gray-400 cursor-not-allowed pointer-events-none"
                }`}
              >
                Open Question in Colab
              </a>

              <a
                href={
                  question.solutionPath
                    ? getColabUrl(question.solutionPath)
                    : undefined
                }
                target="_blank"
                rel="noopener noreferrer"
                className={`flex items-center justify-center gap-2 rounded-2xl px-4 py-3 text-sm font-medium transition shadow-sm ${
                  question.solutionPath
                    ? "bg-lavender-600 text-white hover:bg-lavender-700 active:scale-[0.985]"
                    : "bg-gray-100 text-gray-400 cursor-not-allowed pointer-events-none"
                }`}
              >
                Open Solution in Colab
              </a>
            </div>

            {/* Secondary actions — downloads, matching card subtlety */}
            <div className="grid grid-cols-2 gap-3 mb-3">
              <a
                href={
                  question.questionPath
                    ? getDownloadUrl(question.questionPath)
                    : undefined
                }
                target="_blank"
                rel="noopener noreferrer"
                className={`flex items-center justify-center gap-2 rounded-2xl px-4 py-2 text-sm font-medium border transition ${
                  question.questionPath
                    ? "border-white/60 bg-white/40 text-gray-700 hover:bg-white hover:border-lavender-200"
                    : "border-gray-200 text-gray-400 cursor-not-allowed pointer-events-none"
                }`}
              >
                Download Question
              </a>

              <a
                href={
                  question.solutionPath
                    ? getDownloadUrl(question.solutionPath)
                    : undefined
                }
                target="_blank"
                rel="noopener noreferrer"
                className={`flex items-center justify-center gap-2 rounded-2xl px-4 py-2 text-sm font-medium border transition ${
                  question.solutionPath
                    ? "border-white/60 bg-white/40 text-gray-700 hover:bg-white hover:border-lavender-200"
                    : "border-gray-200 text-gray-400 cursor-not-allowed pointer-events-none"
                }`}
              >
                Download Solution
              </a>
            </div>

          </div>

          <button
            type="button"
            onClick={() =>
              done ? markUndone(question.id) : markDone(question.id)
            }
            className={`mt-4 w-full rounded-lg px-4 py-3 text-sm font-medium transition ${
              done
                ? "border border-gray-300 text-gray-700 hover:bg-gray-50"
                : "bg-lavender-600 text-white hover:bg-lavender-700"
            }`}
          >
            {done ? "Mark as Incomplete" : "Mark as Complete"}
          </button>
        </motion.div>
      </motion.div>
    </motion.div>
  );
}
