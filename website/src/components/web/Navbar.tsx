"use client";

import { useApp } from "@/context/AppContext";
import ProgressMeter from "@/components/shared/ProgressMeter";

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

        {/* Right: Progress + GitHub + Return to Terminal */}
        <div className="flex items-center gap-3">
          <ProgressMeter variant="compact" />
          <a
            href="https://github.com/Exorust/TorchLeet"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-500 hover:text-gray-900 transition"
            aria-label="GitHub"
          >
            <svg viewBox="0 0 16 16" width="22" height="22" fill="currentColor">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
            </svg>
          </a>
          <button
            onClick={() => setMode("terminal")}
            className="bg-gray-900 text-white rounded-full px-5 py-2 text-sm hover:bg-gray-800 transition font-medium"
          >
            Return to Terminal
          </button>
        </div>
      </div>
    </nav>
  );
}
