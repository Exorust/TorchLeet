"use client";

import Image from "next/image";

export default function Footer() {
  return (
    <footer id="about" className="bg-gray-50 border-t border-gray-200 py-12">
      <div className="max-w-6xl mx-auto px-4 flex flex-col items-center gap-6 text-center">
        {/* GitHub star badge */}
        <a
          href="https://github.com/Exorust/TorchLeet"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            src="https://img.shields.io/github/stars/Exorust/TorchLeet?style=social"
            alt="GitHub stars"
            width={100}
            height={20}
            unoptimized
          />
        </a>

        {/* Built by */}
        <p className="text-gray-500 text-sm">
          Built by{" "}
          <a
            href="https://github.com/Exorust"
            target="_blank"
            rel="noopener noreferrer"
            className="text-lavender-600 hover:text-lavender-700 font-medium transition"
          >
            Chandrahas Aroori
          </a>
        </p>

        {/* Feedback */}
        <a
          href="mailto:chandrahas.aroori@gmail.com?subject=TorchLeet%20Feedback"
          className="text-lavender-500 hover:text-lavender-600 text-sm font-medium transition"
        >
          Send Feedback
        </a>

        {/* Tagline */}
        <p className="text-gray-400 text-xs">
          Practice PyTorch. Ace the interview.
        </p>
      </div>
    </footer>
  );
}
