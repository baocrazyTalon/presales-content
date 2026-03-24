import { Redis } from "@upstash/redis";

function getKV() {
  return new Redis({
    url: process.env.UPSTASH_REDIS_REST_URL || "",
    token: process.env.UPSTASH_REDIS_REST_TOKEN || "",
  });
}

const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || "";

function verifyAdmin(req: any): boolean {
  const auth = req.headers.authorization;
  if (!auth || !auth.startsWith("Bearer ")) return false;
  return auth.slice(7) === ADMIN_PASSWORD;
}

export default async function handler(req: any, res: any) {
  try {
    if (req.method !== "GET") {
      return res.status(405).json({ error: "Method not allowed" });
    }

    if (!verifyAdmin(req)) {
      return res.status(401).json({ error: "Unauthorized" });
    }

    const invites: Array<Record<string, unknown>> = [];
    let done = false;
    let cursor = "0";

    const kv = getKV();
    while (!done) {
      const [nextCursor, keys] = (await kv.scan(cursor, {
        match: "invite:*",
        count: 100,
      })) as [string, string[]];

      cursor = nextCursor;
      if (cursor === "0") done = true;

      for (const key of keys) {
        const raw = await kv.get<string>(key);
        if (raw) {
          const invite = typeof raw === "string" ? JSON.parse(raw) : raw;
          const isExpired = invite.expiresAt ? Date.now() > invite.expiresAt : false;
          invites.push({ ...invite, isExpired });
        }
      }
    }

    return res.status(200).json({ invites });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("Invites handler error:", err);
    return res.status(500).json({ error: message });
  }
}
