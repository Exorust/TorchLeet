"use client";

import { motion } from "framer-motion";
import { useApp, type ActiveView } from "@/context/AppContext";
import CompanyFilter from "@/components/shared/CompanyFilter";

const views: { key: ActiveView; label: string; desc: string }[] = [
  {
    key: "basics",
    label: "Basics",
    desc: "Core PyTorch + classical ML foundations",
  },
  {
    key: "llm-path",
    label: "LLM Learning Path",
    desc: "Implement LLM from scratch — curated journey",
  },
  {
    key: "advanced",
    label: "Advanced",
    desc: "Systems, kernels, distributed, modern architectures",
  },
];

export default function PrimaryNavigation() {
  const { activeView, setActiveView } = useApp();

  return (
    <div className="mb-6">
      <div className="flex flex-wrap gap-3 mb-4">
        {views.map((view) => {
          const isActive = activeView === view.key;
          return (
            <button
              key={view.key}
              onClick={() => setActiveView(view.key)}
              className={`group relative rounded-2xl px-5 py-3 text-left transition-all w-full sm:w-auto min-w-[220px] border ${
                isActive
                  ? "border-lavender-600 bg-white shadow-sm"
                  : "border-white/50 bg-white/40 hover:bg-white/70 hover:border-white"
              }`}
            >
              {isActive && (
                <motion.div
                  layoutId="nav-active"
                  className="absolute inset-0 bg-lavender-600/5 rounded-2xl"
                  transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
                />
              )}
              <div className="relative">
                <div
                  className={`font-semibold text-base ${
                    isActive ? "text-lavender-700" : "text-foreground"
                  }`}
                >
                  {view.label}
                </div>
                <div className="text-xs text-foreground/50 mt-0.5 line-clamp-1">
                  {view.desc}
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Company filter is always visible and applies to current view */}
      <CompanyFilter />
    </div>
  );
}
