export default function handler(req: any, res: any) {
  res.status(200).json({ ts_no_import: true, nodeVersion: process.version });
}
