# TubeNote frontend (Next.js). Browser chỉ nói chuyện với container này;
# mọi request /api/* được Next rewrites proxy sang backend.
#
# LƯU Ý: rewrites bị "nướng" vào .next/routes-manifest.json lúc `next build`,
# nên NEXT_PUBLIC_API phải có TRƯỚC bước build (đổi ở runtime không có tác
# dụng) — muốn trỏ backend khác, build lại với build-arg này.
FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG NEXT_PUBLIC_API=http://backend:8010
ENV NEXT_PUBLIC_API=$NEXT_PUBLIC_API
RUN npm run build

EXPOSE 3000
CMD ["npm", "run", "start"]
