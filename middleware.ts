export const config = {
  matcher: ["/test-middleware-only"],
};

export default function middleware() {
  return new Response(JSON.stringify({ ok: true }), {
    headers: { "content-type": "application/json" },
  });
}
