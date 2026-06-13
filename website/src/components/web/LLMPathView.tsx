"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { getLLMPathStages, filterQuestionsByCompanies, type Question } from "@/data/questions";
import { useApp } from "@/context/AppContext";
import QuestionCard from "@/components/web/QuestionCard";
import QuestionModal from "@/components/web/QuestionModal";

export default function LLMPathView() {
  const { companyFilters } = useApp();
  const [selectedQuestion, setSelectedQuestion] = useState<Question | null>(null);

  const stages = getLLMPathStages();

  // Apply company filter to each stage's questions
  const filteredStages = stages.map((stage) => ({
    ...stage,
    questions: filterQuestionsByCompanies(stage.questions, companyFilters),
  })).filter((s) => s.questions.length > 0); // hide empty stages after filter

  const totalInPath = stages.reduce((sum, s) => sum + s.questions.length, 0);
  const visibleCount = filteredStages.reduce((sum, s) => sum + s.questions.length, 0);

  if (stages.length === 0) {
    return <div className="text-foreground/60 py-8">LLM Path content is being curated.</div>;
  }

  return (
    <div className="space-y-10">
      <div className="flex items-baseline justify-between">
        <div>
          <div className="uppercase text-xs tracking-[2px] font-semibold text-lavender-600 mb-1">
            Guided Curriculum
          </div>
          <h2 className="text-2xl font-medium text-foreground">Implement LLM from Scratch</h2>
        </div>
        <div className="text-sm text-foreground/50 font-mono">
          {visibleCount} / {totalInPath} shown
          {companyFilters.length > 0 && " (filtered)"}
        </div>
      </div>

      {filteredStages.length === 0 && companyFilters.length > 0 && (
        <div className="rounded-xl border border-white/60 bg-white/50 p-6 text-center text-sm text-foreground/60">
          No questions in this path match the selected companies. Clear the filter or choose different companies.
        </div>
      )}

      {filteredStages.map((stage, stageIndex) => (
        <div key={stage.stage} className="space-y-3">
          <div className="flex items-center gap-3">
            <div className="font-mono text-sm px-3 py-1 rounded-full bg-lavender-100 text-lavender-700 font-medium">
              {stage.title}
            </div>
            <div className="text-xs text-foreground/40">
              {stage.questions.length} exercise{stage.questions.length === 1 ? "" : "s"}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {stage.questions.map((q, i) => (
              <motion.div
                key={q.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: Math.min(i * 0.02, 0.2) }}
              >
                <QuestionCard
                  question={q}
                  onClick={() => setSelectedQuestion(q)}
                  layoutId={`card-${q.id}`}
                />
              </motion.div>
            ))}
          </div>

          {/* Light visual connector between stages */}
          {stageIndex < filteredStages.length - 1 && (
            <div className="h-px bg-gradient-to-r from-transparent via-lavender-200/60 to-transparent my-2" />
          )}
        </div>
      ))}

      <div className="text-xs text-foreground/50 max-w-prose">
        Follow the stages in order for the best “build an LLM from scratch” experience. Each exercise has a question notebook and a solution. Mark items complete to track progress.
      </div>

      <AnimatePresence>
        {selectedQuestion && (
          <QuestionModal
            key={selectedQuestion.id}
            question={selectedQuestion}
            onClose={() => setSelectedQuestion(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
