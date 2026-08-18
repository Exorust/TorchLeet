import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import {
  generatedProblems,
  getProblemBySlug,
} from "@/data/problems.generated";
import { companySlug, hasCompanyPage } from "@/data/companies";
import Navbar from "@/components/web/Navbar";
import Footer from "@/components/web/Footer";
import DifficultyBadge from "@/components/shared/DifficultyBadge";

const SITE = "https://torch-leet.vercel.app";
const GITHUB = "https://github.com/Exorust/TorchLeet/blob/main";

function displayTitle(title: string): string {
  return /pytorch/i.test(title) ? title : `${title} in PyTorch`;
}

export function generateStaticParams() {
  return generatedProblems.map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const p = getProblemBySlug(slug);
  if (!p) return {};

  const title = `${displayTitle(p.title)} — ML Interview Question`;
  const description =
    p.description ||
    `Implement ${p.title} from scratch in PyTorch. A ${p.difficulty} machine learning interview question with a runnable notebook and solution.`;

  return {
    title,
    description,
    alternates: { canonical: `${SITE}/problems/${p.slug}` },
    openGraph: {
      title,
      description,
      url: `${SITE}/problems/${p.slug}`,
      siteName: "TorchLeet",
      type: "article",
    },
    twitter: { card: "summary_large_image", title, description },
  };
}

export default async function ProblemPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const p = getProblemBySlug(slug);
  if (!p) notFound();

  const ordered = generatedProblems;
  const i = ordered.findIndex((x) => x.slug === p.slug);
  const prev = i > 0 ? ordered[i - 1] : null;
  const next = i < ordered.length - 1 ? ordered[i + 1] : null;

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "LearningResource",
    name: p.title,
    description: p.description,
    educationalLevel: p.difficulty,
    learningResourceType: "Coding exercise",
    teaches: p.title,
    url: `${SITE}/problems/${p.slug}`,
    isAccessibleForFree: true,
    license: "https://opensource.org/licenses/MIT",
  };

  return (
    <div className="min-h-screen bg-lavender-50">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <Navbar />
      <main className="max-w-3xl mx-auto px-4 pt-24 pb-16">
        <nav className="mb-6 text-xs text-foreground/50">
          <Link href="/" className="text-lavender-600 hover:underline">
            torchleet
          </Link>
          <span> / problems / {p.slug}</span>
        </nav>

        <h1 className="mb-4 text-3xl md:text-4xl font-medium text-gray-900 leading-tight">
          {displayTitle(p.title)}
        </h1>

        <div className="mb-6 flex flex-wrap items-center gap-2">
          <DifficultyBadge
            difficulty={
              p.difficulty as "basic" | "easy" | "medium" | "hard" | "expert"
            }
          />
          {p.category ? (
            <span className="bg-lavender-100 text-lavender-600 text-xs rounded-full px-2.5 py-0.5 font-medium">
              {p.category}
            </span>
          ) : null}
          {p.graded ? (
            <span className="bg-green-100 text-green-700 text-xs rounded-full px-2.5 py-0.5 font-medium">
              auto-graded
            </span>
          ) : null}
        </div>

        {p.description ? (
          <p className="mb-8 leading-relaxed text-foreground/70">
            {p.description}
          </p>
        ) : null}

        <section className="mb-8 rounded-2xl border border-gray-200 bg-white/60 backdrop-blur-lg shadow-sm p-5">
          <h2 className="mb-3 text-sm font-semibold text-gray-900">
            Solve it
          </h2>
          <div className="flex flex-wrap gap-3">
            <a
              className="rounded-full bg-lavender-600 px-5 py-2 text-sm font-medium text-white transition hover:bg-lavender-700"
              href={p.colabUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              Open in Colab
            </a>
            <a
              className="rounded-full border border-gray-200 bg-white/70 px-5 py-2 text-sm font-medium text-gray-700 transition hover:border-lavender-200 hover:text-gray-900"
              href={`${GITHUB}/${p.questionPath}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              Question notebook
            </a>
            {p.solutionPath ? (
              <a
                className="rounded-full border border-gray-200 bg-white/70 px-5 py-2 text-sm font-medium text-gray-700 transition hover:border-lavender-200 hover:text-gray-900"
                href={`${GITHUB}/${p.solutionPath}`}
                target="_blank"
                rel="noopener noreferrer"
              >
                Solution
              </a>
            ) : null}
          </div>
        </section>

        {p.graded ? (
          <section className="mb-8 rounded-2xl border border-gray-200 bg-white/60 backdrop-blur-lg shadow-sm p-5">
            <h2 className="mb-2 text-sm font-semibold text-gray-900">
              Check your answer
            </h2>
            <p className="mb-3 text-sm text-foreground/60">
              The grader verifies properties of your implementation, so a
              correct solution written differently from ours still passes.
            </p>
            <pre className="overflow-x-auto rounded-xl bg-terminal-bg p-4 font-mono text-xs text-terminal-text">
              {`pip install torchleet

from torchleet import check
check("${p.slug}"${p.entries.map((e) => `, ${e}`).join("")})`}
            </pre>
          </section>
        ) : null}

        {p.companies.length ? (
          <section className="mb-8">
            <h2 className="mb-3 text-sm font-semibold text-gray-900">
              Company tags
            </h2>
            <div className="flex flex-wrap gap-2">
              {p.companies.map((c) => (
                <Link
                  key={c}
                  href={`/company/${companySlug(c)}`}
                  className="bg-lavender-100 text-lavender-600 text-xs rounded-full px-2.5 py-0.5 font-medium transition hover:bg-lavender-200"
                >
                  {c}
                  {p.companyConfidence[c] === "reported" && (
                    <span className="opacity-60"> · reported</span>
                  )}
                </Link>
              ))}
            </div>
            <p className="mt-3 text-xs text-foreground/50">
              <Link href="/#sources" className="text-lavender-600 hover:underline">
                How these tags are sourced
              </Link>
            </p>
          </section>
        ) : null}

        <nav className="mt-12 flex justify-between border-t border-gray-200 pt-6 text-sm">
          {prev ? (
            <Link
              href={`/problems/${prev.slug}`}
              className="text-lavender-600 hover:text-lavender-700 font-medium transition"
            >
              ← {prev.title}
            </Link>
          ) : (
            <span />
          )}
          {next ? (
            <Link
              href={`/problems/${next.slug}`}
              className="text-lavender-600 hover:text-lavender-700 font-medium transition"
            >
              {next.title} →
            </Link>
          ) : (
            <span />
          )}
        </nav>
      </main>
      <Footer />
    </div>
  );
}
