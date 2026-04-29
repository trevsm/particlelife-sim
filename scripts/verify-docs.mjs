/**
 * Ensures docs/index.html only references assets that exist under docs/.
 * Prevents GitHub Pages black screens when HTML is deployed without matching hashed bundles.
 */
import fs from "fs"
import path from "path"
import { fileURLToPath } from "url"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const docsDir = path.join(__dirname, "..", "docs")
const indexPath = path.join(docsDir, "index.html")

if (!fs.existsSync(indexPath)) {
  console.error("verify-docs: missing docs/index.html — run npm run build")
  process.exit(1)
}

const html = fs.readFileSync(indexPath, "utf8")
const urls = []
for (const m of html.matchAll(/\s(?:src|href)="([^"]+)"/g)) {
  urls.push(m[1])
}

let failed = false
for (const url of urls) {
  if (url.startsWith("data:")) continue

  let rel
  if (url.startsWith("/particlelife-sim/")) {
    rel = url.slice("/particlelife-sim/".length)
  } else if (url.startsWith("./")) {
    rel = url.slice(2)
  } else if (!url.startsWith("/")) {
    rel = url
  } else {
    console.warn(
      "verify-docs: skip check for root-absolute URL (not under base):",
      url
    )
    continue
  }

  const fp = path.join(docsDir, rel)
  if (!fs.existsSync(fp)) {
    console.error("verify-docs: missing file for", url)
    console.error("           expected at", fp)
    failed = true
  }
}

if (failed) {
  console.error(
    "verify-docs: FAIL — deploy the full docs/ folder together after each build."
  )
  process.exit(1)
}

console.log("verify-docs: OK (all referenced assets present under docs/)")
