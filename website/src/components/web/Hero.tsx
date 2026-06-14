"use client";

import { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import Link from "next/link";
import ProgressMeter from "@/components/shared/ProgressMeter";

export default function Hero() {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end start"],
  });

  const y = useTransform(scrollYProgress, [0, 1], ["0%", "-12%"]);
  const opacity = useTransform(scrollYProgress, [0, 0.8], [1, 0]);

  return (
    <section ref={ref} className="pt-10 md:pt-16 pb-4 md:pb-6">
      <motion.div
        style={{ y, opacity }}
        className="flex flex-col lg:flex-row items-start gap-8"
      >
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="flex-1 space-y-4"
        >
          <div className="flex items-center gap-4">
            <h1 className="text-4xl md:text-5xl font-medium text-lavender-600 leading-tight">
              Grind PyTorch for ML/AI Interviews
            </h1>
          </div>

          <p className="text-lg text-foreground/60 leading-relaxed max-w-xl">
            Follow the guided <span className="font-medium text-foreground/70">LLM Learning Path</span> to implement a model from scratch, browse curated <span className="font-medium text-foreground/70">Basics</span> and <span className="font-medium text-foreground/70">Advanced</span> lists, or set up the <span className="font-medium text-foreground/70">AI Tutor</span> for interactive coaching. Many exercises are tagged with the real companies that ask them in interviews.
          </p>

          <div className="flex items-center gap-4 flex-wrap">
            <a
              href="https://github.com/Exorust/TorchLeet"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="https://img.shields.io/github/stars/Exorust/TorchLeet?style=social"
                alt="GitHub stars"
                height="20"
              />
            </a>
            <Link
              href="/ai-tutor"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-lavender-600 text-white text-sm font-medium hover:bg-lavender-700 transition-colors shadow-sm"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2a4 4 0 0 1 4 4v2a4 4 0 0 1-8 0V6a4 4 0 0 1 4-4Z" />
                <path d="M18 14a6.97 6.97 0 0 0-4-1.7" />
                <path d="M6 14a6.97 6.97 0 0 1 4-1.7" />
                <rect x="2" y="14" width="20" height="8" rx="4" />
              </svg>
              Set Up AI Tutor
            </Link>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2, ease: "easeOut" }}
          className="w-full lg:w-[440px] shrink-0"
        >
          <ProgressMeter />
        </motion.div>
      </motion.div>
    </section>
  );
}
