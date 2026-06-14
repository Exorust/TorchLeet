"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

const DISMISSED_KEY = "torchleet-ai-tutor-banner-dismissed";

export default function AnnouncementBanner() {
  const [dismissed, setDismissed] = useState(true);

  useEffect(() => {
    setDismissed(localStorage.getItem(DISMISSED_KEY) === "1");
  }, []);

  if (dismissed) return null;

  return (
    <div className="relative mb-8 rounded-2xl border border-lavender-200 bg-gradient-to-r from-lavender-50 via-white to-lavender-50 p-5 shadow-sm overflow-hidden">
      {/* Shimmer accent */}
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-lavender-400 to-transparent animate-pulse" />

      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-3 min-w-0">
          <span className="shrink-0 text-xs font-bold uppercase tracking-wider bg-lavender-600 text-white px-2.5 py-1 rounded-full">
            New
          </span>
          <div className="min-w-0">
            <div className="font-semibold text-sm text-foreground">
              AI Tutor is here
            </div>
            <div className="text-xs text-foreground/50 mt-0.5">
              Turn any AI assistant into your PyTorch coach. One command, no install.
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <Link
            href="/ai-tutor"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-lavender-600 text-white text-sm font-medium hover:bg-lavender-700 transition-colors shadow-sm"
          >
            Learn more
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </Link>
          <button
            onClick={() => {
              localStorage.setItem(DISMISSED_KEY, "1");
              setDismissed(true);
            }}
            className="text-foreground/30 hover:text-foreground/60 transition p-1"
            aria-label="Dismiss"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
