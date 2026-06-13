"use client";

import { useState } from "react";
import { AnimatePresence } from "framer-motion";
import { getAdvancedGroupedByTopic, filterQuestionsByCompanies, type Question } from "@/data/questions";
import { useApp } from "@/context/AppContext";
import QuestionCard from "@/components/web/QuestionCard";
import QuestionModal from "@/components/web/QuestionModal";

export default function AdvancedGroupedView() {
  const { companyFilters } = useApp();
  const [selectedQuestion, setSelectedQuestion] = useState<Question | null>(null);

  const topics = getAdvancedGroupedByTopic();

  // Apply company filter to each topic's questions (like LLM Path does)
  const filteredTopics = topics
    .map((topic) => ({
      ...topic,
      questions: filterQuestionsByCompanies(topic.questions, companyFilters),
    }))
    .filter((t) => t.questions.length > 0);

  const total = topics.reduce((sum, t) => sum + t.questions.length, 0);
  const visible = filteredTopics.reduce((sum, t) => sum + t.questions.length, 0);

  if (topics.length === 0) {
    return <div className="text-foreground/60 py-8">Advanced content is being curated.</div>;
  }

  return (
    <div className="space-y-10">
      <div className="flex items-baseline justify-between">
        <div>
          <div className="uppercase text-xs tracking-[2px] font-semibold text-foreground/50 mb-1">
            Topic Grouped
          </div>
          <h2 className="text-2xl font-medium text-foreground">Advanced List</h2>
        </div>
        <div className="text-sm text-foreground/50 font-mono">
          {visible} / {total} shown
          {companyFilters.length > 0 && " (filtered)"}
        </div>
      </div>

      {filteredTopics.length === 0 && companyFilters.length > 0 && (
        <div className="rounded-xl border border-white/60 bg-white/50 p-6 text-center text-sm text-foreground/60">
          No advanced questions match the selected companies. Clear the filter to see all topics.
        </div>
      )}

      {filteredTopics.map((topic) => (
        <div key={topic.topic} className="space-y-3">
          <div className="flex items-center gap-3">
            <div className="font-medium text-base px-3 py-1 rounded-full bg-white/70 border border-white/60 text-foreground/80">
              {topic.title}
            </div>
            <div className="text-xs text-foreground/40">
              {topic.questions.length} exercise{topic.questions.length === 1 ? "" : "s"}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {topic.questions.map((q) => (
              <QuestionCard
                key={q.id}
                question={q}
                onClick={() => setSelectedQuestion(q)}
                layoutId={`card-${q.id}`}
              />
            ))}
          </div>
        </div>
      ))}

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
