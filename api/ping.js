export default function handler(req, res) {
  res.status(200).json({ pong: true, time: new Date().toISOString() });
}
