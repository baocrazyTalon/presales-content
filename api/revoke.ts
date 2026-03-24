import type { VercelRequest, VercelResponse } from "@vercel/node";
import { Redis } from "@upstash/redis";

function getKV() {
  return new Redis({
    url: process.env.UPSTASH_REDIS_REST_URL || "",
    token: process.env.UPSTASH_REDIS_REST_TOKEN || "",
  });
}

const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || "";

function verifyAdmin(req: VercelRequest): boolean {
  const auth = req.headers.authorization;
  if (!auth || !auth.startsWith("Bearer ")) return false;
  return auth.slice(7) === ADMIN_PASSWORD;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  if (!verifyAdmin(req)) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  const { token } = req.body as { token: string };
  if (!token) {
    return res.status(400).json({ error: "token is required" });
  }

  // Get invite to find file path for cleaning up the index key
  const kv = getKV();
  const inviteRaw = await kv.get<string>(`invite:${token}`);
  if (inviteRaw) {
    const invite = typeof inviteRaw === "string" ? JSON.parse(inviteRaw) : inviteRaw;
    const fileKey = `file_invite:${Buffer.from(invite.filePath).toString("base64url")}:${token}`;
    await kv.del(fileKey);
  }

  // Delete the main invite record
  await kv.del(`invite:${token}`);

  return res.status(200).json({ success: true });
}
