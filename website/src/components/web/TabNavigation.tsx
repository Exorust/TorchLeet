"use client";

interface Props {
  activeSet: "v1" | "v2" | "v3";
  onSetChange: (set: "v1" | "v2" | "v3") => void;
}

const tabs: { key: "v1" | "v2" | "v3"; label: string; isNew?: boolean }[] = [
  { key: "v1", label: "Question Set (v1)" },
  { key: "v2", label: "LLM Set (v2)" },
  { key: "v3", label: "Advanced ML Systems (v3)", isNew: true },
];

export default function TabNavigation({ activeSet, onSetChange }: Props) {
  return (
    <div className="flex flex-wrap gap-3 mb-6">
      {tabs.map((tab) => (
        <button
          key={tab.key}
          onClick={() => onSetChange(tab.key)}
          className={`rounded-full px-6 py-2 text-sm font-medium transition ${
            activeSet === tab.key
              ? "bg-lavender-500 text-white"
              : "bg-lavender-100 text-lavender-700 hover:bg-lavender-200"
          }`}
        >
          {tab.label}
          {tab.isNew && (
            <span className="ml-2 bg-red-500 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full uppercase">
              New
            </span>
          )}
        </button>
      ))}
    </div>
  );
}
