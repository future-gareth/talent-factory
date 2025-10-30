import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow cross-origin access from talentfactory.local
  allowedDevOrigins: ['http://talentfactory.local', 'http://localhost:3004'],
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8084/:path*",
      },
    ];
  },
  async headers() {
    // Relax CSP to allow Google Fonts and backend websocket/API connections
    const csp = [
      "default-src 'self'",
      "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
      "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
      "style-src-elem 'self' 'unsafe-inline' https://fonts.googleapis.com",
      "font-src 'self' data: https://fonts.gstatic.com",
      "img-src 'self' data: blob:",
      // Allow API + HMR over localhost and talentfactory.local through nginx
      "connect-src 'self' http://localhost:8084 ws://localhost:8084 http://localhost:3200 ws://localhost:3200 http://talentfactory.local ws://talentfactory.local",
      "frame-ancestors 'self'",
    ].join("; ");

    return [
      {
        source: "/(.*)",
        headers: [
          { key: "Content-Security-Policy", value: csp },
          { key: "Referrer-Policy", value: "no-referrer" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "SAMEORIGIN" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
        ],
      },
    ];
  },
};

export default nextConfig;
