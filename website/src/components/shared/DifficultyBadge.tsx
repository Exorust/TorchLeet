"use client";

interface Props {
  difficulty: "basic" | "easy" | "medium" | "hard" | "expert";
}

const colorMap: Record<Props["difficulty"], string> = {
  basic: "bg-blue-100 text-blue-700",
  easy: "bg-green-100 text-green-700",
  medium: "bg-yellow-100 text-yellow-800",
  hard: "bg-red-100 text-red-700",
  expert: "bg-purple-100 text-purple-700",
};

export default function DifficultyBadge({ difficulty }: Props) {
  return (
    <span
      className={`text-xs font-medium px-2.5 py-0.5 rounded-full ${colorMap[difficulty]}`}
    >
      {difficulty.charAt(0).toUpperCase() + difficulty.slice(1)}
    </span>
  );
}
