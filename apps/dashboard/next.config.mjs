/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The trading engine serves the API in-process. In development the dashboard
  // runs on its own port, so requests are proxied rather than hard-coding the
  // engine's origin into the client bundle.
  async rewrites() {
    const api = process.env.KAT_API_URL ?? "http://127.0.0.1:8000";
    return [{ source: "/api/:path*", destination: `${api}/api/:path*` }];
  },
};
export default nextConfig;
