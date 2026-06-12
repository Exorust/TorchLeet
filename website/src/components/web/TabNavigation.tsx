"use client";

import { motion } from "framer-motion";

interface Props {
  activeSet: "v1" | "v2" | "v3";
  onSetChange: (set: "v1" | "v2" | "v3") => void;
}

const tabs: { key: "v1" | "v2" | "v3"; label: string }[] = [
  { key: "v1", label: "Question Set (v1)" },
  { key: "v2", label: "LLM Set (v2)" },
  { key: "v3", label: "Advanced ML Systems (v3)" },
];

export default function TabNavigation({ activeSet, onSetChange }: Props) {
  return (
    <div className="flex flex-wrap gap-3 mb-6">
      {tabs.map((tab) => (
        <button
          key={tab.key}
          onClick={() => onSetChange(tab.key)}
          className={`relative rounded-full px-6 py-2 text-sm font-medium transition-colors ${
            activeSet === tab.key
              ? "text-white"
              : "text-foreground/60 hover:text-foreground"
          }`}
        >
          {activeSet === tab.key && (
            <motion.div
              layoutId="tab-pill"
              className="absolute inset-0 bg-lavender-600 rounded-full shadow-sm"
              transition={{ type: "spring", bounce: 0.15, duration: 0.5 }}
            />
          )}
          {activeSet !== tab.key && (
            <div className="absolute inset-0 bg-white/50 backdrop-blur-md border border-white/50 rounded-full" />
          )}
          <span className="relative z-10">{tab.label}</span>
        </button>
      ))}
    </div>
  );
}
