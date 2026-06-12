"use client";

import { useState } from "react";
import { motion, AnimatePresence, LayoutGroup } from "framer-motion";
import { type Question } from "@/data/questions";
import QuestionCard from "@/components/web/QuestionCard";
import QuestionModal from "@/components/web/QuestionModal";

interface Props {
  questions: Question[];
}

const cardVariants = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: {
      delay: Math.min(i * 0.05, 0.3),
      duration: 0.4,
      ease: "easeOut" as const,
    },
  }),
};

export default function QuestionGrid({ questions }: Props) {
  const [selectedQuestion, setSelectedQuestion] = useState<Question | null>(
    null,
  );

  return (
    <LayoutGroup>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {questions.map((question, i) => (
          <motion.div
            key={question.id}
            custom={i}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-30px" }}
            variants={cardVariants}
            layout="position"
          >
            <QuestionCard
              question={question}
              layoutId={`card-${question.id}`}
              onClick={() => setSelectedQuestion(question)}
            />
          </motion.div>
        ))}
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
    </LayoutGroup>
  );
}
