"use client";

import { motion, useScroll, useSpring } from "framer-motion";
import { useApp } from "@/context/AppContext";
import { getQuestionsBySet, getQuestionsByCategory } from "@/data/questions";
import Navbar from "@/components/web/Navbar";
import Hero from "@/components/web/Hero";
import TabNavigation from "@/components/web/TabNavigation";
import V3SubTabs from "@/components/web/V3SubTabs";
import QuestionGrid from "@/components/web/QuestionGrid";
import Footer from "@/components/web/Footer";
import ScrollReveal from "@/components/shared/ScrollReveal";

function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, {
    stiffness: 100,
    damping: 30,
    restDelta: 0.001,
  });

  return (
    <motion.div
      className="fixed top-0 left-0 right-0 h-[3px] bg-lavender-600 z-50 origin-left"
      style={{ scaleX }}
    />
  );
}

export default function WebMode() {
  const { activeSet, setActiveSet, activeV3Category, setActiveV3Category } =
    useApp();

  const filteredQuestions =
    activeSet === "v3" && activeV3Category !== null
      ? getQuestionsByCategory(activeV3Category)
      : getQuestionsBySet(activeSet);

  return (
    <div className="min-h-screen bg-lavender-50">
      <ScrollProgress />
      <Navbar />
      <main className="max-w-6xl mx-auto px-4 pt-24">
        <Hero />
        <section id="questions" className="py-8">
          <ScrollReveal>
            <TabNavigation activeSet={activeSet} onSetChange={setActiveSet} />
          </ScrollReveal>
          {activeSet === "v3" && (
            <ScrollReveal delay={0.1}>
              <V3SubTabs
                activeCategory={activeV3Category}
                onCategoryChange={setActiveV3Category}
              />
            </ScrollReveal>
          )}
          <QuestionGrid questions={filteredQuestions} />
        </section>
      </main>
      <Footer />
    </div>
  );
}
