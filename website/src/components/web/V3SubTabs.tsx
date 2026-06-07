"use client";

import { type V3Category } from "@/data/questions";

interface Props {
  activeCategory: V3Category | null;
  onCategoryChange: (cat: V3Category | null) => void;
}

const categories: { key: V3Category | null; label: string }[] = [
  { key: null, label: "All" },
  { key: "classical-ml", label: "Classical ML" },
  { key: "llm-decoding", label: "LLM Decoding" },
  { key: "llm-inference", label: "LLM Inference" },
  { key: "modern-architectures", label: "Modern Architectures" },
  { key: "alignment-training", label: "Alignment & Training" },
  { key: "gpu-systems", label: "GPU Systems" },
];

export default function V3SubTabs({ activeCategory, onCategoryChange }: Props) {
  return (
    <div className="flex flex-wrap gap-2 mb-6">
      {categories.map((cat) => (
        <button
          key={cat.key ?? "all"}
          onClick={() => onCategoryChange(cat.key)}
          className={`text-sm px-4 py-1.5 rounded-full transition font-medium ${
            activeCategory === cat.key
              ? "text-lavender-600 bg-lavender-50 border-b-2 border-lavender-600"
              : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
          }`}
        >
          {cat.label}
        </button>
      ))}
    </div>
  );
}
