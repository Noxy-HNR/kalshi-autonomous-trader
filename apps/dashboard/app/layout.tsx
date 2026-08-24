import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/nav";

export const metadata: Metadata = {
  title: "Kalshi Autonomous Trader",
  description:
    "Control and monitoring for autonomous Kalshi event-contract and perpetual trading. Perpetual leverage is capped at 1.00x.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <aside className="sidebar">
            <div className="brand">
              Kalshi Trader
              <small>max perp leverage 1.00&times;</small>
            </div>
            <Nav />
          </aside>
          <main className="main">{children}</main>
        </div>
      </body>
    </html>
  );
}
