/** @type {import('next').NextConfig} */
const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8010";

const nextConfig = {
  async rewrites() {
    // Proxy /api/* sang FastAPI để tránh CORS lúc dev.
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
  images: { remotePatterns: [{ protocol: "https", hostname: "**" }] },
};

export default nextConfig;
