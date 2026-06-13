"use client";

import { motion, useScroll, useSpring } from "framer-motion";
import { useApp } from "@/context/AppContext";
import {
  getBasicsQuestions,
  getAdvancedQuestions,
  getLLMPathQuestions,
  filterQuestionsByCompanies,
} from "@/data/questions";
import Navbar from "@/components/web/Navbar";
import Hero from "@/components/web/Hero";
import PrimaryNavigation from "@/components/web/PrimaryNavigation";
import QuestionGrid from "@/components/web/QuestionGrid";
import LLMPathView from "@/components/web/LLMPathView";
import AdvancedGroupedView from "@/components/web/AdvancedGroupedView";
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
  const { activeView, companyFilters } = useApp();

  // Compute base list for the chosen view, then apply company filters
  let baseQuestions: ReturnType<typeof getBasicsQuestions> = [];
  let isPathView = false;

  if (activeView === "llm-path") {
    isPathView = true;
  } else if (activeView === "basics") {
    baseQuestions = getBasicsQuestions();
  } else {
    baseQuestions = getAdvancedQuestions();
  }

  const filteredForGrid = filterQuestionsByCompanies(baseQuestions, companyFilters);

  return (
    <div className="min-h-screen bg-lavender-50">
      <ScrollProgress />
      <Navbar />
      <main className="max-w-6xl mx-auto px-4 pt-24">
        <Hero />
        <section id="questions" className="py-8">
          <ScrollReveal>
            <PrimaryNavigation />
          </ScrollReveal>

          {isPathView ? (
            <LLMPathView />
          ) : activeView === "advanced" ? (
            <AdvancedGroupedView />
          ) : (
            <QuestionGrid questions={filteredForGrid} />
          )}
        </section>
      </main>
      <Footer />
    </div>
  );
}
