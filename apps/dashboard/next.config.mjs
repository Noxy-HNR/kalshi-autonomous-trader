/** @type {import('next').NextConfig} */

/**
 * There is deliberately no `rewrites()` entry for `/api/*`.
 *
 * Next.js resolves rewrites at build time and freezes them into the route
 * manifest, which made the engine's origin a property of whoever ran
 * `next build` rather than of the machine the dashboard runs on. Proxying is
 * done by the route handler in `app/api/[...path]/route.ts`, which reads the
 * environment per request.
 */
const nextConfig = {
  reactStrictMode: true,
};
export default nextConfig;
