import type { Metadata } from "next";
import { JetBrains_Mono, Inter } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import { AppProvider } from "@/context/AppContext";
import { ProgressProvider } from "@/context/ProgressContext";
import "performative-ui/styles.css";
import "./globals.css";

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  weight: ["400", "500", "700"],
});

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "TorchLeet - PyTorch Interview Prep",
  description:
    "LLM Learning Path + Basics & Advanced lists. Practice real PyTorch & ML systems interview questions. Filter by company. Build models from scratch.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${jetbrainsMono.variable} ${inter.variable} h-full`}>
      <body className="h-full font-sans">
        <AppProvider>
          <ProgressProvider>{children}</ProgressProvider>
        </AppProvider>
        <Analytics />
      </body>
    </html>
  );
}
