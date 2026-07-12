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
              {/* Waveform — giọng nói là sản phẩm; bar nhấp nhô khi hover (CSS) */}
              <svg viewBox="0 0 24 24" fill="currentColor">
                <rect x="2.5" y="8.5" width="3" height="7" rx="1.5" />
                <rect x="8" y="5" width="3" height="14" rx="1.5" />
                <rect x="13.5" y="2.5" width="3" height="19" rx="1.5" />
                <rect x="19" y="7" width="3" height="10" rx="1.5" />
              </svg>
            </span>
            <span className="brand">TuBeNote</span>
            <NavLinks />
          </div>
          <div className="topbar-actions">
            <NavLinks variant="create" />
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
