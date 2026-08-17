import type { Metadata } from "next";
import Link from "next/link";
import {
  getCompanies,
  getCompaniesWithPages,
  COMPANY_PAGE_MIN_PROBLEMS,
} from "@/data/companies";
import { generatedProblems } from "@/data/problems.generated";
import Navbar from "@/components/web/Navbar";
import Footer from "@/components/web/Footer";
import CompanyGrid from "@/components/web/CompanyGrid";

const SITE = "https://torch-leet.vercel.app";

export function generateMetadata(): Metadata {
  const title = "Company-wise PyTorch interview problems — TorchLeet";
  const description =
    "Browse machine learning and PyTorch interview problems by the company that asks them — Anthropic, OpenAI, Google, Meta, NVIDIA and more.";
  return {
    title,
    description,
    alternates: { canonical: `${SITE}/company` },
    openGraph: {
      title,
      description,
      url: `${SITE}/company`,
      siteName: "TorchLeet",
      type: "website",
    },
    twitter: { card: "summary_large_image", title, description },
  };
}

export default function CompanyIndexPage() {
  // Only companies above the threshold get pages, so only they get links here.
  const companies = getCompaniesWithPages();
  const smaller = getCompanies().filter(
    (c) => c.count < COMPANY_PAGE_MIN_PROBLEMS,
  );
  const taggedProblems = generatedProblems.filter(
    (p) => p.companies.length > 0,
  ).length;

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: "Company-wise PyTorch interview problems",
    itemListElement: companies.map((c, i) => ({
      "@type": "ListItem",
      position: i + 1,
      name: c.name,
      url: `${SITE}/company/${c.slug}`,
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
          <p className="text-xs font-semibold uppercase tracking-wider text-foreground/50 mb-2">
            Companies
          </p>
          <h1 className="text-4xl md:text-5xl font-medium text-lavender-600 leading-tight mb-4">
            Company-wise interview problems
          </h1>
          <p className="text-foreground/60 max-w-2xl mb-2">
            PyTorch and ML interview problems tagged with the companies whose
            interview loops and job posts touch these topics — pick a company
            and grind its list.
          </p>
          <p className="text-xs text-foreground/50 mb-8">
            {companies.length} companies · {taggedProblems} tagged problems ·{" "}
            <Link href="/#sources" className="text-lavender-600 hover:underline">
              How these tags are sourced
            </Link>
          </p>

          <CompanyGrid companies={companies} />

          {smaller.length > 0 ? (
            <div className="mt-10">
              <p className="text-xs text-foreground/50 mb-2">
                Also tagged, with fewer than {COMPANY_PAGE_MIN_PROBLEMS} problems
                so far:
              </p>
              <p className="text-sm text-foreground/60">
                {smaller.map((c) => `${c.name} (${c.count})`).join(" · ")}
              </p>
            </div>
          ) : null}
        </section>
      </main>
      <Footer />
    </div>
  );
}
