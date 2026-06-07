"use client";

import { useState } from "react";
import { type Question } from "@/data/questions";
import QuestionCard from "@/components/web/QuestionCard";
import QuestionModal from "@/components/web/QuestionModal";

interface Props {
  questions: Question[];
}

export default function QuestionGrid({ questions }: Props) {
  const [selectedQuestion, setSelectedQuestion] = useState<Question | null>(
    null,
  );

  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {questions.map((question) => (
          <QuestionCard
            key={question.id}
            question={question}
            onClick={() => setSelectedQuestion(question)}
          />
        ))}
      </div>

      <QuestionModal
        question={selectedQuestion}
        onClose={() => setSelectedQuestion(null)}
      />
    </>
  );
}
