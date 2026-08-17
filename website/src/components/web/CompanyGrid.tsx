"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { difficultyBreakdown, type CompanySummary } from "@/data/companies";
import { useProgress } from "@/context/ProgressContext";

interface Props {
  companies: CompanySummary[];
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

function CompanyCard({ company }: { company: CompanySummary }) {
  const { isDone } = useProgress();
  const done = company.problemIds.filter((id) => isDone(id)).length;
  const percent =
    company.count > 0 ? Math.round((done / company.count) * 100) : 0;

  return (
    <Link
      href={`/company/${company.slug}`}
      className="block h-full rounded-2xl border border-gray-200 bg-white/60 backdrop-blur-lg shadow-sm p-5 transition-all hover:shadow-md hover:border-lavender-200"
    >
      {/* Top row */}
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-medium text-base text-gray-900">{company.name}</h3>
        {company.hasReported && (
          <span className="bg-lavender-100 text-lavender-600 text-xs rounded-full px-2 py-0.5 font-medium">
            reported
          </span>
        )}
      </div>

      {/* Count + difficulty breakdown */}
      <div className="mb-4">
        <span className="text-2xl font-bold text-lavender-600">
          {company.count}
        </span>{" "}
        <span className="text-sm text-gray-500">
          problem{company.count === 1 ? "" : "s"}
        </span>
        <p className="text-xs text-gray-500 mt-1">
          {difficultyBreakdown(company.byDifficulty)}
        </p>
      </div>

      {/* Progress */}
      <div>
        <div className="flex items-center justify-between text-[10px] text-gray-500 mb-1">
          <span>
            {done}/{company.count} solved
          </span>
          <span>{percent}%</span>
        </div>
        <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-lavender-600 rounded-full transition-all duration-500"
            style={{ width: `${percent}%` }}
          />
        </div>
      </div>
    </Link>
  );
}

export default function CompanyGrid({ companies }: Props) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {companies.map((company, i) => (
        <motion.div
          key={company.slug}
          custom={i}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-30px" }}
          variants={cardVariants}
          layout="position"
        >
          <CompanyCard company={company} />
        </motion.div>
      ))}
    </div>
  );
}
