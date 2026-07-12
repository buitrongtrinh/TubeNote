"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

// Nav tách 2 loại: NƠI CHỐN (Thư viện/Bản nháp — trái, cạnh logo) và HÀNH
// ĐỘNG tạo mới (CTA cam bên phải, kiểu "+ New" của các SaaS). Cùng component,
// render 2 chỗ trong topbar qua prop variant.
const PLACES = [
  { href: "/", label: "Thư viện" },
  { href: "/drafts", label: "Bản nháp" },
];

export default function NavLinks({ variant = "places" }) {
  const path = usePathname();

  if (variant === "create") {
    const active = path.startsWith("/add");
    return (
      <Link href="/add" className={"create-cta" + (active ? " active" : "")}>
        <span aria-hidden="true">＋</span> Tạo lồng tiếng
      </Link>
    );
  }

  return (
    <nav className="nav-actions">
      {PLACES.map((l) => {
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
