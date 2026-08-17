"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import type { GeneratedProblem } from "@/data/problems.generated";
import DifficultyBadge from "@/components/shared/DifficultyBadge";

interface Props {
  companyName: string;
  problems: GeneratedProblem[];
}

const rowVariants = {
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

export default function CompanyProblemList({ companyName, problems }: Props) {
  return (
    <div className="flex flex-col gap-3">
      {problems.map((p, i) => (
        <motion.div
          key={p.slug}
          custom={i}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-30px" }}
          variants={rowVariants}
          layout="position"
        >
          <Link
            href={`/problems/${p.slug}`}
            className="flex items-center justify-between gap-4 rounded-2xl border border-gray-200 bg-white/60 backdrop-blur-lg shadow-sm px-5 py-4 transition-all hover:shadow-md hover:border-lavender-200"
          >
            <div className="min-w-0">
              <h3 className="font-medium text-base text-gray-900 line-clamp-1">
                {p.title}
              </h3>
              {p.description && (
                <p className="text-gray-500 text-sm line-clamp-1">
                  {p.description}
                </p>
              )}
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <DifficultyBadge
                difficulty={p.difficulty as "basic" | "easy" | "medium" | "hard" | "expert"}
              />
              {p.companyConfidence[companyName] === "reported" && (
                <span className="bg-lavender-100 text-lavender-600 text-xs rounded-full px-2 py-0.5 font-medium">
                  reported
                </span>
              )}
            </div>
          </Link>
        </motion.div>
      ))}
    </div>
  );
}
