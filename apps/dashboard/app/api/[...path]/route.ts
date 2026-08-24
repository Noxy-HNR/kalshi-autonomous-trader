/**
 * Runtime proxy to the trading engine's API.
 *
 * This exists instead of a `rewrites()` entry because Next.js resolves rewrites
 * at **build** time and freezes them into the route manifest. That made the
 * engine's origin a property of whoever ran `next build`, so a deployment that
 * moved the engine to another port kept proxying to the old one and rendered
 * "cannot reach the trading engine" on every page, with the real destination
 * visible nowhere in the configuration.
 *
 * A route handler reads the environment on each request, so one setting -- the
 * same KAT_API_HOST / KAT_API_PORT the engine itself reads -- governs both
 * sides, at the moment it is actually used.
 */

import { NextRequest } from "next/server";

// Never prerendered or cached: this is a live view of a system trading money.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

function engineOrigin(): string {
  if (process.env.KAT_API_URL) return process.env.KAT_API_URL.replace(/\/+$/, "");
  const host = process.env.KAT_API_HOST ?? "127.0.0.1";
  const port = process.env.KAT_API_PORT ?? "8000";
  return `http://${host}:${port}`;
}

async function proxy(request: NextRequest, path: string[]): Promise<Response> {
  const target = `${engineOrigin()}/api/${path.join("/")}${request.nextUrl.search}`;

  let response: Response;
  try {
    response = await fetch(target, {
      method: request.method,
      headers: { "Content-Type": request.headers.get("content-type") ?? "application/json" },
      body: request.method === "GET" || request.method === "HEAD"
        ? undefined
        : await request.text(),
      cache: "no-store",
    });
  } catch (error) {
    // The engine being down is an ordinary, expected state -- it is a separate
    // process. Say so precisely, including where we looked, rather than
    // returning an opaque 500 that sends the reader to the wrong place.
    return Response.json(
      {
        detail:
          `Cannot reach the trading engine at ${engineOrigin()}. ` +
          `Start it with \`python -m kat.engine.entrypoint\`, or set KAT_API_HOST / ` +
          `KAT_API_PORT if it is listening elsewhere. (${String(error)})`,
      },
      { status: 502 },
    );
  }

  const body = await response.text();
  return new Response(body, {
    status: response.status,
    headers: { "Content-Type": response.headers.get("content-type") ?? "application/json" },
  });
}

type Context = { params: Promise<{ path: string[] }> };

export async function GET(request: NextRequest, context: Context) {
  return proxy(request, (await context.params).path);
}

export async function POST(request: NextRequest, context: Context) {
  return proxy(request, (await context.params).path);
}
