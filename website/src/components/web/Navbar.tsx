"use client";

import { useApp } from "@/context/AppContext";

export default function Navbar() {
  const { setMode } = useApp();

  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-40 bg-white/95 backdrop-blur-sm shadow-sm">
      <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
        {/* Left: Logo */}
        <span className="font-mono font-bold text-lavender-700 text-lg tracking-tight select-none">
          TORCHLEET
        </span>

        {/* Center: Nav links */}
        <div className="hidden sm:flex items-center gap-8">
          <button
            onClick={() => scrollTo("questions")}
            className="text-sm text-gray-600 hover:text-lavender-700 transition font-medium"
          >
            Questions
          </button>
          <button
            onClick={() => scrollTo("about")}
            className="text-sm text-gray-600 hover:text-lavender-700 transition font-medium"
          >
            About
          </button>
        </div>

        {/* Right: Return to Terminal */}
        <button
          onClick={() => setMode("terminal")}
          className="bg-gray-900 text-white rounded-full px-5 py-2 text-sm hover:bg-gray-800 transition font-medium"
        >
          Return to Terminal
        </button>
      </div>
    </nav>
  );
}
