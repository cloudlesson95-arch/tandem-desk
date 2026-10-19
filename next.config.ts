import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // In development the Python API runs under uvicorn; on Vercel, vercel.json routes it.
  async rewrites() {
    if (process.env.NODE_ENV !== "development") return [];
    return [
      {
        source: "/api/agent/:path*",
        destination: "http://127.0.0.1:8000/api/agent/:path*",
      },
    ];
  },
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
