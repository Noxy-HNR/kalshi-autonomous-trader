"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const SECTIONS: Array<{ group: string; items: Array<[string, string]> }> = [
  {
    group: "Monitor",
    items: [
      ["/", "Overview"],
      ["/events", "Event Contracts"],
      ["/perps", "Perpetuals"],
      ["/positions", "Positions"],
      ["/orders", "Orders"],
      ["/markets", "Markets"],
    ],
  },
  {
    group: "Research",
    items: [
      ["/strategies", "Strategies"],
      ["/discovery", "Strategy Discovery"],
      ["/models", "Models"],
      ["/arbitrage", "Arbitrage"],
      ["/news", "News & Research"],
    ],
  },
  {
    group: "Operate",
    items: [
      ["/risk", "Risk"],
      ["/controls", "Controls"],
      ["/system", "System"],
      ["/audit", "Audit Log"],
    ],
  },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <nav className="nav">
      {SECTIONS.map((section) => (
        <div key={section.group}>
          <div className="group">{section.group}</div>
          {section.items.map(([href, label]) => (
            <Link key={href} href={href} className={pathname === href ? "active" : ""}>
              {label}
            </Link>
          ))}
        </div>
      ))}
    </nav>
  );
}
