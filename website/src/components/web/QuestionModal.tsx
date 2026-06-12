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

  const setBadgeColor: Record<string, string> = {
    v1: "bg-gray-100 text-gray-600",
    v2: "bg-blue-50 text-blue-600",
    v3: "bg-lavender-100 text-lavender-600",
  };

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
              <span
                className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase ${setBadgeColor[question.set]}`}
              >
                {question.set}
              </span>
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

          {/* Action buttons 2x2 grid */}
          <div className="grid grid-cols-2 gap-3">
            <a
              href={
                question.questionPath
                  ? getColabUrl(question.questionPath)
                  : undefined
              }
              target="_blank"
              rel="noopener noreferrer"
              className={`flex items-center justify-center gap-2 rounded-lg px-4 py-3 text-sm font-medium transition ${
                question.questionPath
                  ? "bg-lavender-600 text-white hover:bg-lavender-700"
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
              className={`flex items-center justify-center gap-2 rounded-lg px-4 py-3 text-sm font-medium transition ${
                question.solutionPath
                  ? "bg-lavender-600 text-white hover:bg-lavender-700"
                  : "bg-gray-100 text-gray-400 cursor-not-allowed pointer-events-none"
              }`}
            >
              Open Solution in Colab
            </a>

            <a
              href={
                question.questionPath
                  ? getDownloadUrl(question.questionPath)
                  : undefined
              }
              target="_blank"
              rel="noopener noreferrer"
              className={`flex items-center justify-center gap-2 rounded-lg px-4 py-3 text-sm font-medium border transition ${
                question.questionPath
                  ? "border-gray-300 text-gray-700 hover:bg-gray-50"
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
              className={`flex items-center justify-center gap-2 rounded-lg px-4 py-3 text-sm font-medium border transition ${
                question.solutionPath
                  ? "border-gray-300 text-gray-700 hover:bg-gray-50"
                  : "border-gray-200 text-gray-400 cursor-not-allowed pointer-events-none"
              }`}
            >
              Download Solution
            </a>
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
