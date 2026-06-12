"use client";

import { type Question } from "@/data/questions";
import { useProgress } from "@/context/ProgressContext";
import type { MeterDifficulty } from "@/lib/progress";

interface Props {
  questions?: Question[];
  variant?: "full" | "compact";
}

const DIFFICULTY_COLORS: Record<MeterDifficulty, string> = {
  easy: "bg-green-500",
  medium: "bg-yellow-500",
  hard: "bg-red-500",
};

const DIFFICULTY_LABELS: Record<MeterDifficulty, string> = {
  easy: "Easy",
  medium: "Medium",
  hard: "Hard",
};

function CircularProgress({
  percent,
  completed,
  total,
  size = 96,
}: {
  percent: number;
  completed: number;
  total: number;
  size?: number;
}) {
  const stroke = 8;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percent / 100) * circumference;

  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={stroke}
          className="text-gray-200"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="text-lavender-600 transition-all duration-500"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        {size <= 40 ? (
          <span className="text-[8px] font-bold text-gray-700 leading-none">
            {percent}%
          </span>
        ) : (
          <>
            <span className="text-lg font-bold text-gray-900 leading-none">
              {completed}
            </span>
            <span className="text-[10px] text-gray-500 mt-0.5">/ {total}</span>
          </>
        )}
      </div>
    </div>
  );
}

export default function ProgressMeter({
  questions,
  variant = "full",
}: Props) {
  const { getStats } = useProgress();
  const stats = getStats(questions);

  if (variant === "compact") {
    return (
      <div
        className="flex items-center gap-2 text-sm text-gray-600"
        title={`${stats.completed} of ${stats.total} solved (${stats.percent}%)`}
      >
        <CircularProgress
          percent={stats.percent}
          completed={stats.completed}
          total={stats.total}
          size={36}
        />
        <span className="hidden md:inline font-medium">
          {stats.completed}/{stats.total}
        </span>
      </div>
    );
  }

  return (
    <div className="bg-white/60 backdrop-blur-lg rounded-2xl shadow-sm border border-white/50 p-7">
      <div className="flex items-center gap-8">
        <CircularProgress
          percent={stats.percent}
          completed={stats.completed}
          total={stats.total}
          size={110}
        />
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-gray-900 mb-1">
            Solved {stats.completed} of {stats.total}
          </h3>
          <p className="text-xs text-gray-500 mb-4">
            {stats.percent}% complete
            {questions && questions.length < stats.total
              ? " in this view"
              : ""}
          </p>
          <div className="space-y-2.5">
            {(["easy", "medium", "hard"] as MeterDifficulty[]).map((level) => {
              const bucket = stats.byDifficulty[level];
              return (
                <div key={level}>
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="font-medium text-gray-700">
                      {DIFFICULTY_LABELS[level]}
                    </span>
                    <span className="text-gray-500">
                      {bucket.done}/{bucket.total}
                    </span>
                  </div>
                  <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${DIFFICULTY_COLORS[level]}`}
                      style={{ width: `${bucket.percent}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
