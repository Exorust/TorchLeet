import type { Metadata } from "next";
import { JetBrains_Mono } from "next/font/google";
import { AppProvider } from "@/context/AppContext";
import "./globals.css";

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  weight: ["400", "500", "700"],
});

export const metadata: Metadata = {
  title: "TorchLeet - PyTorch Interview Prep",
  description:
    "Practice PyTorch problems for ML/AI interviews. 79 questions from basic to expert, tagged with real interview companies.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${jetbrainsMono.variable} h-full`}>
      <body className="h-full font-mono">
        <AppProvider>{children}</AppProvider>
      </body>
    </html>
  );
}
