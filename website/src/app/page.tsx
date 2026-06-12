"use client";

import { useApp } from "@/context/AppContext";
import Terminal from "@/components/terminal/Terminal";
import WebMode from "@/components/web/WebMode";
import FeedbackPopup from "@/components/shared/FeedbackPopup";
import GitHubCorner from "@/components/shared/GitHubCorner";

export default function Home() {
  const { mode } = useApp();

  return (
    <main className="relative w-screen h-screen overflow-hidden">
      <div
        className={`absolute inset-0 transition-opacity duration-300 ${
          mode === "terminal"
            ? "opacity-100 z-10"
            : "opacity-0 z-0 pointer-events-none"
        }`}
      >
        <Terminal />
      </div>
      <div
        className={`absolute inset-0 transition-opacity duration-300 overflow-y-auto ${
          mode === "web"
            ? "opacity-100 z-10"
            : "opacity-0 z-0 pointer-events-none"
        }`}
      >
        <WebMode />
      </div>
      {mode === "terminal" && <GitHubCorner />}
      <FeedbackPopup />
    </main>
  );
}
