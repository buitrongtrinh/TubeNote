import "./globals.css";
import NavLinks from "@/components/NavLinks";
import ThemeToggle from "@/components/ThemeToggle";

export const metadata = {
  title: "TuBeNote — Auto Dubbing",
  description: "Lồng tiếng video YouTube tự động",
};

// Đặt theme trước khi paint để không chớp trắng/đen (no-FOUC). Chạy inline
// ngay đầu <body>, trước khi React hydrate.
const themeInitScript = `(function(){try{var t=localStorage.getItem('tubenote-theme');if(t!=='light'&&t!=='dark'){t=window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}document.documentElement.dataset.theme=t;}catch(e){document.documentElement.dataset.theme='light';}})();`;

export default function RootLayout({ children }) {
  return (
    <html lang="vi" suppressHydrationWarning>
      <body>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
        <header className="topbar">
          <div className="topbar-left">
            <span className="brand-mark" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M8.2 5.4v13.2c0 .8.9 1.3 1.6.9l10.4-6.6c.6-.4.6-1.4 0-1.8L9.8 4.5c-.7-.4-1.6.1-1.6.9Z" />
              </svg>
            </span>
            <span className="brand">TuBeNote</span>
          </div>
          <div className="topbar-actions">
            <NavLinks />
            <ThemeToggle />
          </div>
        </header>
        <div className="app-shell">
          {children}
        </div>
      </body>
    </html>
  );
}
