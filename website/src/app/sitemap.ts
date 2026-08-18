import type { MetadataRoute } from "next";
import { generatedProblems } from "@/data/problems.generated";
import { getCompaniesWithPages } from "@/data/companies";

const SITE = "https://torch-leet.vercel.app";

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  return [
    { url: SITE, lastModified: now, changeFrequency: "weekly", priority: 1 },
    { url: `${SITE}/ai-tutor`, lastModified: now, changeFrequency: "monthly", priority: 0.7 },
    { url: `${SITE}/company`, lastModified: now, changeFrequency: "weekly", priority: 0.7 },
    ...getCompaniesWithPages().map((c) => ({
      url: `${SITE}/company/${c.slug}`,
      lastModified: now,
      changeFrequency: "weekly" as const,
      priority: 0.6,
    })),
    ...generatedProblems.map((p) => ({
      url: `${SITE}/problems/${p.slug}`,
      lastModified: now,
      changeFrequency: "monthly" as const,
      priority: 0.8,
    })),
  ];
}
