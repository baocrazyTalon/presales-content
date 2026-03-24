import { Redis } from "@upstash/redis";
import { Resend } from "resend";
import { randomUUID } from "crypto";

function getKV() {
  return new Redis({
    url: process.env.UPSTASH_REDIS_REST_URL || "",
    token: process.env.UPSTASH_REDIS_REST_TOKEN || "",
  });
}

const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || "";
const BASE_URL = process.env.BASE_URL || "";
const RESEND_API_KEY = process.env.RESEND_API_KEY || "";
const SENDER_EMAIL = process.env.SENDER_EMAIL || "onboarding@resend.dev";

function verifyAdmin(req: any): boolean {
  const auth = req.headers.authorization;
  if (!auth || !auth.startsWith("Bearer ")) return false;
  return auth.slice(7) === ADMIN_PASSWORD;
}

export default async function handler(req: any, res: any) {
  try {
    if (req.method !== "POST") {
      return res.status(405).json({ error: "Method not allowed" });
    }

    if (!verifyAdmin(req)) {
      return res.status(401).json({ error: "Unauthorized" });
    }

    const { email, filePath, expiryDays } = req.body as {
      email: string;
      filePath: string;
      expiryDays?: number | null;
    };

    if (!email || !filePath) {
      return res.status(400).json({ error: "email and filePath are required" });
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({ error: "Invalid email format" });
    }

    const token = randomUUID();
    const now = Date.now();
    const expiresAt = expiryDays ? now + expiryDays * 24 * 60 * 60 * 1000 : null;

    const invite = {
      token,
      email,
      filePath,
      expiresAt,
      createdAt: now,
    };

    // Store invite in KV
    const kv = getKV();
    await kv.set(`invite:${token}`, JSON.stringify(invite));

    if (expiryDays) {
      await kv.expire(`invite:${token}`, expiryDays * 24 * 60 * 60);
    }

    // Index by file for listing
    const fileKey = `file_invite:${Buffer.from(filePath).toString("base64url")}:${token}`;
    await kv.set(fileKey, token);
    if (expiryDays) {
      await kv.expire(fileKey, expiryDays * 24 * 60 * 60);
    }

    // Send email
    const accessUrl = `${BASE_URL}/api/access?token=${token}`;
    const resend = new Resend(RESEND_API_KEY);

    try {
      await resend.emails.send({
        from: SENDER_EMAIL,
        to: email,
        subject: "You've been invited to view a document",
        html: `
          <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 520px; margin: 0 auto; padding: 32px;">
            <h2 style="margin-bottom: 8px;">Document Access Invitation</h2>
            <p style="color: #555; line-height: 1.6;">
              You've been granted access to view a document. Click the button below to open it.
              ${expiryDays ? `<br/><strong>This link expires in ${expiryDays} day${expiryDays > 1 ? "s" : ""}.</strong>` : ""}
            </p>
            <a href="${accessUrl}" style="display: inline-block; margin-top: 16px; padding: 12px 28px; background: #4361ee; color: #fff; border-radius: 6px; text-decoration: none; font-weight: 600;">
              Open Document
            </a>
            <p style="margin-top: 24px; font-size: 0.85rem; color: #999;">
              If the button doesn't work, copy and paste this URL into your browser:<br/>
              <a href="${accessUrl}" style="color: #4361ee;">${accessUrl}</a>
            </p>
          </div>
        `,
      });
    } catch (emailError) {
      console.error("Failed to send email:", emailError);
      return res.status(200).json({
        success: true,
        token,
        warning: "Invite created but email delivery failed. Share the link manually.",
        accessUrl,
      });
    }

    return res.status(200).json({ success: true, token, accessUrl });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("Invite handler error:", err);
    return res.status(500).json({ error: message });
  }
}
