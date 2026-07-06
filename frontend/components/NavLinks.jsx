"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Thư viện" },
  { href: "/drafts", label: "Bản nháp" },
  { href: "/add", label: "Tạo lồng tiếng" },
];

export default function NavLinks() {
  const path = usePathname();
  return (
    <nav className="nav-actions">
      {LINKS.map((l) => {
        const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
        return (
          <Link key={l.href} href={l.href} className={active ? "active" : ""}>
            {l.label}
          </Link>
        );
      })}
    </nav>
  );
}
