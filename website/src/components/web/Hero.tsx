"use client";

import Image from "next/image";

const stats = [
  { value: "35", label: "PyTorch Questions" },
  { value: "25", label: "LLM Questions" },
  { value: "30", label: "Advanced ML Questions" },
];

export default function Hero() {
  return (
    <section className="py-16 md:py-24">
      <div className="flex flex-col md:flex-row items-center gap-10">
        {/* Text content */}
        <div className="flex-1 space-y-6">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 leading-tight">
            Grind PyTorch for ML/AI Interviews
          </h1>
          <p className="text-lg text-gray-600 leading-relaxed max-w-xl">
            Practice problems from basic PyTorch to advanced LLM systems. Tagged
            with real interview companies. TorchLeet V3 is now out with 30 new
            questions and company-wise filtering!
          </p>

          {/* Stats row */}
          <div className="flex flex-wrap gap-4 pt-4">
            {stats.map((stat) => (
              <div
                key={stat.label}
                className="bg-white rounded-xl shadow-sm border border-lavender-100 px-6 py-4 text-center"
              >
                <div className="text-3xl font-bold text-lavender-700">
                  {stat.value}
                </div>
                <div className="text-xs text-gray-500 mt-1 font-medium">
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Hero image (desktop only) */}
        <div className="hidden md:block flex-shrink-0">
          <Image
            src="/torchleet-llm.png"
            alt="TorchLeet mascot"
            width={320}
            height={320}
            unoptimized
            className="rounded-2xl"
            priority
          />
        </div>
      </div>
    </section>
  );
}
