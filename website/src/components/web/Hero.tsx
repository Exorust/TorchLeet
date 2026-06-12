"use client";

import { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
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
    <section ref={ref} className="py-10 md:py-16">
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
            Practice problems from basic PyTorch to advanced LLM systems. Tagged
            with real interview companies. TorchLeet V3 is now out with 30 new
            questions and company-wise filtering!
          </p>

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
