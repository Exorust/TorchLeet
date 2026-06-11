"use client";

import { useApp } from "@/context/AppContext";
import { getQuestionsBySet, getQuestionsByCategory } from "@/data/questions";
import Navbar from "@/components/web/Navbar";
import Hero from "@/components/web/Hero";
import TabNavigation from "@/components/web/TabNavigation";
import V3SubTabs from "@/components/web/V3SubTabs";
import QuestionGrid from "@/components/web/QuestionGrid";
import Footer from "@/components/web/Footer";
import ProgressMeter from "@/components/shared/ProgressMeter";

export default function WebMode() {
  const { activeSet, setActiveSet, activeV3Category, setActiveV3Category } =
    useApp();

  const filteredQuestions =
    activeSet === "v3" && activeV3Category !== null
      ? getQuestionsByCategory(activeV3Category)
      : getQuestionsBySet(activeSet);

  return (
    <div className="min-h-screen bg-gradient-to-b from-lavender-50 to-white">
      <Navbar />
      <main className="max-w-6xl mx-auto px-4 pt-20">
        <Hero />
        <section id="questions" className="py-12">
          <TabNavigation activeSet={activeSet} onSetChange={setActiveSet} />
          {activeSet === "v3" && (
            <V3SubTabs
              activeCategory={activeV3Category}
              onCategoryChange={setActiveV3Category}
            />
          )}
          <div className="mb-6">
            <ProgressMeter questions={filteredQuestions} />
          </div>
          <QuestionGrid questions={filteredQuestions} />
        </section>
      </main>
      <Footer />
    </div>
  );
}
