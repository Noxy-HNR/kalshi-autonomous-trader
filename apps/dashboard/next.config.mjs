/** @type {import('next').NextConfig} */

/**
 * Where the trading engine serves its API.
 *
 * The engine reads KAT_API_HOST and KAT_API_PORT. The dashboard used to read a
 * separate KAT_API_URL defaulting to port 8000, so changing the engine's port
 * left the dashboard proxying to a port nothing was listening on -- every page
 * rendered "cannot reach the trading engine" with a 500, and the two settings
 * that had to agree did not even share a name.
 *
 * One source of truth now: the same variables the engine reads, with KAT_API_URL
 * still honoured as an explicit override for the case where the engine genuinely
 * lives somewhere else.
 */
function engineOrigin() {
  if (process.env.KAT_API_URL) return process.env.KAT_API_URL;
  const host = process.env.KAT_API_HOST ?? "127.0.0.1";
  const port = process.env.KAT_API_PORT ?? "8000";
  return `http://${host}:${port}`;
}

const nextConfig = {
  reactStrictMode: true,
  // The trading engine serves the API in-process. The dashboard runs on its own
  // port, so requests are proxied rather than hard-coding the engine's origin
  // into the client bundle.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${engineOrigin()}/api/:path*` }];
  },
};
export default nextConfig;
