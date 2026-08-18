import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getCompaniesWithPages, getCompanyBySlug } from "@/data/companies";
import Navbar from "@/components/web/Navbar";
import Footer from "@/components/web/Footer";
import CompanyProblemList from "@/components/web/CompanyProblemList";
import DifficultyBadge from "@/components/shared/DifficultyBadge";

const SITE = "https://torch-leet.vercel.app";

const DIFFICULTY_ORDER = ["basic", "easy", "medium", "hard", "expert"] as const;

export function generateStaticParams() {
  return getCompaniesWithPages().map((c) => ({ name: c.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ name: string }>;
}): Promise<Metadata> {
  const { name } = await params;
  const company = getCompanyBySlug(name);
  if (!company) return {};

  const title = `${company.name} PyTorch interview problems — TorchLeet`;
  const description = `${company.problems.length} machine learning and PyTorch interview problems reported by candidates at ${company.name}, with runnable notebooks and solutions.`;

  return {
    title,
    description,
    alternates: { canonical: `${SITE}/company/${company.slug}` },
    openGraph: {
      title,
      description,
      url: `${SITE}/company/${company.slug}`,
      siteName: "TorchLeet",
      type: "website",
    },
    twitter: { card: "summary_large_image", title, description },
  };
}

export default async function CompanyPage({
  params,
}: {
  params: Promise<{ name: string }>;
}) {
  const { name } = await params;
  const company = getCompanyBySlug(name);
  if (!company) notFound();

  const byDifficulty: Record<string, number> = {};
  for (const p of company.problems) {
    byDifficulty[p.difficulty] = (byDifficulty[p.difficulty] ?? 0) + 1;
  }

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `${company.name} PyTorch interview problems`,
    itemListElement: company.problems.map((p, i) => ({
      "@type": "ListItem",
      position: i + 1,
      item: {
        "@type": "LearningResource",
        name: p.title,
        description: p.description,
        educationalLevel: p.difficulty,
        learningResourceType: "Coding exercise",
        url: `${SITE}/problems/${p.slug}`,
        isAccessibleForFree: true,
        license: "https://opensource.org/licenses/MIT",
      },
    })),
  };

  return (
    <div className="min-h-screen bg-lavender-50">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <Navbar />
      <main className="max-w-6xl mx-auto px-4 pt-24 pb-16">
        <section className="py-8">
          <Link
            href="/company"
            className="text-sm text-lavender-600 hover:text-lavender-700 font-medium transition"
          >
            ← All companies
          </Link>

          <div className="mt-4 mb-8">
            <p className="text-xs font-semibold uppercase tracking-wider text-foreground/50 mb-2">
              Company
            </p>
            <h1 className="text-4xl md:text-5xl font-medium text-lavender-600 leading-tight mb-4">
              {company.name}
            </h1>
            <p className="text-foreground/60 max-w-2xl mb-3">
              {company.problems.length} problem
              {company.problems.length === 1 ? "" : "s"} tagged for{" "}
              {company.name}.
            </p>
            <div className="flex flex-wrap items-center gap-2">
              {DIFFICULTY_ORDER.filter((d) => byDifficulty[d]).map((d) => (
                <span key={d} className="flex items-center gap-1">
                  <DifficultyBadge difficulty={d} />
                  <span className="text-xs text-foreground/50">
                    × {byDifficulty[d]}
                  </span>
                </span>
              ))}
              <span className="text-xs text-foreground/50 ml-2">
                <Link
                  href="/#sources"
                  className="text-lavender-600 hover:underline"
                >
                  How these tags are sourced
                </Link>
              </span>
            </div>
          </div>

          <CompanyProblemList
            companyName={company.name}
            problems={company.problems}
          />
        </section>
      </main>
      <Footer />
    </div>
  );
}
