import { Redis } from "@upstash/redis";
import { SignJWT } from "jose";

function getKV() {
  return new Redis({
    url: process.env.UPSTASH_REDIS_REST_URL || "",
    token: process.env.UPSTASH_REDIS_REST_TOKEN || "",
  });
}

const JWT_SECRET = new TextEncoder().encode(process.env.JWT_SECRET || "");

export default async function handler(req: any, res: any) {
  if (req.method !== "GET") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const token = req.query.token as string;
  if (!token) {
    return res.redirect(302, "/denied.html");
  }

  // Look up invite in KV
  const kv = getKV();
  const inviteRaw = await kv.get<string>(`invite:${token}`);
  if (!inviteRaw) {
    return res.redirect(302, "/denied.html");
  }

  const invite = typeof inviteRaw === "string" ? JSON.parse(inviteRaw) : inviteRaw;

  // Check expiry
  if (invite.expiresAt && Date.now() > invite.expiresAt) {
    return res.redirect(302, "/denied.html");
  }

  // Create a signed JWT cookie scoped to this file
  const jwt = await new SignJWT({
    filePath: invite.filePath,
    inviteToken: invite.token,
    email: invite.email,
  })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime(invite.expiresAt ? new Date(invite.expiresAt) : "365d")
    .sign(JWT_SECRET);

  // Cookie name is unique per file path
  const cookieName = `access_${Buffer.from(invite.filePath).toString("base64url")}`;

  // Set cookie and redirect to the file
  const maxAge = invite.expiresAt
    ? Math.floor((invite.expiresAt - Date.now()) / 1000)
    : 365 * 24 * 60 * 60;

  res.setHeader(
    "Set-Cookie",
    `${cookieName}=${jwt}; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=${maxAge}`
  );

  return res.redirect(302, `/${invite.filePath}`);
}
