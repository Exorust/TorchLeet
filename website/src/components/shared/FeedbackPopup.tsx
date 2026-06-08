"use client";

import { useState, useEffect } from "react";

const DELAY_MS = 5 * 60 * 1000;
const DISMISSED_KEY = "torchleet-feedback-dismissed";

export default function FeedbackPopup() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (sessionStorage.getItem(DISMISSED_KEY)) return;
    const timer = setTimeout(() => setVisible(true), DELAY_MS);
    return () => clearTimeout(timer);
  }, []);

  function dismiss() {
    setVisible(false);
    sessionStorage.setItem(DISMISSED_KEY, "1");
  }

  if (!visible) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-xl max-w-sm w-full mx-4 p-6 text-center">
        <p className="text-lg font-bold text-gray-900 mb-2">
          Share feedback with the dev?
        </p>
        <p className="text-sm text-gray-500 mb-6">
          Bug reports, question suggestions, interview stories - I read everything.
        </p>
        <div className="flex gap-3">
          <button
            onClick={dismiss}
            className="flex-1 px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200 transition"
          >
            Maybe later
          </button>
          <a
            href="mailto:chandrahas.aroori@gmail.com?subject=TorchLeet%20Feedback"
            onClick={dismiss}
            className="flex-1 px-4 py-2 text-sm font-medium text-white bg-purple-600 rounded-lg hover:bg-purple-700 transition text-center"
          >
            Send feedback
          </a>
        </div>
      </div>
    </div>
  );
}
