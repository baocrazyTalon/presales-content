import { next, rewrite } from "@vercel/edge";
import { jwtVerify } from "jose";

export const config = {
  matcher: [
    "/((?!api/|index\\.html$|admin\\.html$|denied\\.html$).*\\.html$)",
    "/my-muscle-chef",
  ],
};

/** Base64url encode a string (Edge Runtime compatible — no Buffer) */
function toBase64Url(str: string): string {
  return btoa(String.fromCharCode(...new TextEncoder().encode(str)))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

const JWT_SECRET = new TextEncoder().encode(process.env.JWT_SECRET || "");

/** Minimal KV GET for Edge Runtime (no Node.js deps) */
async function kvGet(key: string): Promise<string | null> {
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;
  if (!url || !token) return null;
  const res = await fetch(`${url}/get/${encodeURIComponent(key)}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) return null;
  const data = await res.json();
  return data.result ?? null;
}

export default async function middleware(request: Request) {
  const reqUrl = new URL(request.url);
  const filePath = decodeURIComponent(reqUrl.pathname);

  // Normalize: strip leading slash for cookie name consistency
  const normalizedPath = filePath.startsWith("/") ? filePath.slice(1) : filePath;

  // Look for an access cookie for this specific file
  const cookies = request.headers.get("cookie") || "";
  const cookieName = `access_${toBase64Url(normalizedPath)}`;
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
