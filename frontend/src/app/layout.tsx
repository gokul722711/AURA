import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AURA — Autonomous Research & Engineering Agent",
  description:
    "An LLM-agnostic agentic AI platform for complex research and engineering tasks.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
