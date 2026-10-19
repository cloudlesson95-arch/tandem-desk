import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Tandem Desk",
  description:
    "A guarded multi-agent support assistant for a fictional bike-rental service: agentic RAG, tool calls with human approval, PII redaction, guardrails, long-term memory and a pre-display fact check.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col bg-zinc-50 font-sans text-zinc-900 dark:bg-black dark:text-zinc-100">
        <header className="border-b border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
          <nav className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-3">
            <Link href="/" className="font-semibold">
              Tandem Desk
              <span className="ml-2 text-xs font-normal text-zinc-500">support for Velomarsh, a fictional bike rental</span>
            </Link>
            <div className="ml-auto flex gap-4 text-sm">
              <Link href="/" className="hover:text-teal-700 dark:hover:text-teal-300">
                Chat
              </Link>
              <Link href="/eval" className="hover:text-teal-700 dark:hover:text-teal-300">
                Eval
              </Link>
            </div>
          </nav>
        </header>
        {children}
      </body>
    </html>
  );
}
