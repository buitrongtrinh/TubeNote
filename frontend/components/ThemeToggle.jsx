"use client";

import { useEffect, useState } from "react";

const STORAGE_KEY = "tubenote-theme";

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <circle cx="12" cy="12" r="4.4" />
      <path d="M12 2.5v2.4M12 19.1v2.4M2.5 12h2.4M19.1 12h2.4M5 5l1.7 1.7M17.3 17.3 19 19M19 5l-1.7 1.7M6.7 17.3 5 19" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor">
      <path d="M20.6 14.6A8.6 8.6 0 0 1 9.4 3.4a.55.55 0 0 0-.7-.7A9.7 9.7 0 1 0 21.3 15.3a.55.55 0 0 0-.7-.7Z" />
    </svg>
  );
}

export default function ThemeToggle() {
  // Chỉ đọc theme sau khi mount để icon SSR/CSR khớp nhau (theme nằm ở
  // localStorage, server không biết trước).
  const [theme, setTheme] = useState(null);

  useEffect(() => {
    setTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light");
  }, []);

  function toggle() {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* chặn private mode ném lỗi quota */
    }
    setTheme(next);
  }

  const dark = theme === "dark";
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={toggle}
      title={dark ? "Chuyển giao diện sáng" : "Chuyển giao diện tối"}
      aria-label={dark ? "Chuyển giao diện sáng" : "Chuyển giao diện tối"}
    >
      {theme === null ? null : dark ? <SunIcon /> : <MoonIcon />}
    </button>
  );
}
