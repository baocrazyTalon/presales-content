import { next, rewrite } from "@vercel/edge";
import { jwtVerify } from "jose";

export const config = {
  matcher: [
    /*
     * Match all .html requests EXCEPT:
     *  - /index.html (public hub)
     *  - /admin.html (has its own auth)
     *  - /denied.html (must be accessible)
     *  - /api/* routes
     */
    "/((?!api/|index\\.html$|admin\\.html$|denied\\.html$).*\\.html$)",
    // Also match rewritten paths (e.g. /my-muscle-chef)
    "/my-muscle-chef",
  ],
};

const UPSTASH_URL = process.env.UPSTASH_REDIS_REST_URL!;
const UPSTASH_TOKEN = process.env.UPSTASH_REDIS_REST_TOKEN!;
const JWT_SECRET = new TextEncoder().encode(process.env.JWT_SECRET || "");

/** Minimal KV GET for Edge Runtime (no Node.js deps) */
async function kvGet(key: string): Promise<string | null> {
  const res = await fetch(`${UPSTASH_URL}/get/${encodeURIComponent(key)}`, {
    headers: { Authorization: `Bearer ${UPSTASH_TOKEN}` },
  });
  if (!res.ok) return null;
  const data = await res.json();
  return data.result ?? null;
}

export default async function middleware(request: Request) {
  const url = new URL(request.url);
  let filePath = decodeURIComponent(url.pathname);

  // Normalize: strip leading slash for cookie name consistency
  const normalizedPath = filePath.startsWith("/") ? filePath.slice(1) : filePath;

  // Look for an access cookie for this specific file
  const cookies = request.headers.get("cookie") || "";
  const cookieName = `access_${Buffer.from(normalizedPath).toString("base64url")}`;
  const cookieMatch = cookies
    .split(";")
    .map((c) => c.trim())
    .find((c) => c.startsWith(`${cookieName}=`));

  if (!cookieMatch) {
    return rewrite(new URL("/denied.html", request.url));
  }

  const token = cookieMatch.split("=")[1];

  // Verify JWT signature
  try {
    const { payload } = await jwtVerify(token, JWT_SECRET);

    // Check the JWT is for this specific file
    if (payload.filePath !== normalizedPath) {
      return rewrite(new URL("/denied.html", request.url));
    }

    // Verify the invite still exists in KV and is not expired
    const inviteRaw = await kvGet(`invite:${payload.inviteToken}`);
    if (!inviteRaw) {
      return rewrite(new URL("/denied.html", request.url));
    }

    const invite = JSON.parse(inviteRaw);

    // Check expiry
    if (invite.expiresAt && Date.now() > invite.expiresAt) {
      return rewrite(new URL("/denied.html", request.url));
    }

    // All checks passed — allow through
    return next();
  } catch {
    return rewrite(new URL("/denied.html", request.url));
  }
}
