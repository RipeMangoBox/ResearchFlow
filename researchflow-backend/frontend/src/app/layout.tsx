import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
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
  title: "ResearchFlow",
  description: "Web-first research operating system",
};

const navItems = [
  { href: "/", label: "Dashboard" },
  { href: "/papers", label: "Papers" },
  { href: "/import", label: "Import" },
  { href: "/search", label: "Search" },
  { href: "/bottlenecks", label: "Bottlenecks" },
  { href: "/lineage", label: "Lineage" },
  { href: "/graph", label: "Graph" },
  { href: "/reviews", label: "Reviews" },
  { href: "/reports", label: "Reports" },
  { href: "/digests", label: "Digests" },
  { href: "/directions", label: "Directions" },
];

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="zh"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-gray-50">
        <nav className="bg-white border-b px-6 py-3 flex items-center gap-6 shrink-0">
          <Link href="/" className="font-bold text-lg text-gray-900">
            ResearchFlow
          </Link>
          <div className="flex gap-4">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
              >
                {item.label}
              </Link>
            ))}
          </div>
        </nav>
        <main className="flex-1 p-6">{children}</main>
      </body>
    </html>
  );
}
