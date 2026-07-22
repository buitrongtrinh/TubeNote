/** @type {import('next').NextConfig} */
const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8010";

const nextConfig = {
  async rewrites() {
    // Proxy /api/* sang FastAPI để tránh CORS lúc dev.
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
  images: { remotePatterns: [{ protocol: "https", hostname: "**" }] },
  experimental: {
    // Luồng video /api/stream đi qua proxy rewrite này. Browser xem video tải
    // theo chu kỳ đầy-buffer-thì-ngừng-đọc 30-60s; proxyTimeout mặc định 30s
    // (xem next/dist/server/lib/router-utils/proxy-request.js: "we limit proxy
    // requests to 30s by default") giết ngầm kết nối upstream trong lúc đó ->
    // browser rút cạn ~10MB đệm sót rồi đứng hình ở phút 2-4 (đã tái hiện + cô
    // lập bằng test: backend trực tiếp sống sau 90s im, qua proxy chết).
    //
    // Đây là timeout NHÀN RỖI trên socket upstream, mà phát video thì nhàn rỗi
    // dài là hành vi bình thường -> mọi ngưỡng hữu hạn đều chỉ đẩy lỗi ra xa
    // hơn chứ không hết. 24h = dài hơn mọi phiên xem thực tế, kể cả video dài
    // bị tạm dừng rồi bỏ đó (trần 30 phút trước đây vẫn chạm được).
    //
    // ĐỪNG đặt 0 để hòng "tắt timeout": dòng `proxyTimeout || 30000` coi 0 là
    // falsy nên rơi ngược về đúng 30s. Chỉ `null` mới tắt hẳn, nhưng schema là
    // z.number() nên null trượt validate -> dùng số lớn là đường duy nhất.
    proxyTimeout: 86_400_000,
  },
};

export default nextConfig;
