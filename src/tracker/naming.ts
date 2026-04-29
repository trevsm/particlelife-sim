/**
 * Organism naming.
 *
 * Two names are produced from the type histogram:
 *
 * 1. **Engineering signature** — terse, encodes top-2 types and a 4-hex hash:
 *    e.g. `T0:62-T2:23#A3F1`. Live; updates as composition shifts.
 *
 * 2. **Colloquial name** — phonetic, syllables drawn from per-type pools and
 *    composed by the dominant types: e.g. `Karonvi`. Birth-frozen so the
 *    organism keeps a stable identity even as composition drifts slightly.
 *
 * Both are deterministic in the histogram (after quantization to 5% bins for
 * the hash), so identical "species" get matching names.
 */

const SYLLABLES: string[][] = [
  ["ka", "drak", "ron"],
  ["lu", "sol", "ja"],
  ["vi", "rell", "syl"],
  ["cy", "mer", "thal"],
  ["xa", "vorn", "psy"],
  ["rho", "nim", "fae"],
  ["mor", "tau", "gren"],
  ["zep", "iri", "ven"],
  ["ash", "kor", "lyn"],
  ["um", "bri", "sael"],
  ["ory", "tal", "wen"],
  ["fyr", "esh", "vel"],
  ["nul", "ord", "hyx"],
  ["pelt", "rho", "ary"],
  ["zin", "cre", "myr"],
  ["ob", "lyx", "tas"],
]

const FALLBACK_PRE = ["zu", "ty", "ka", "li", "no", "vi", "ru", "se", "an", "fe"]
const FALLBACK_POST = ["mor", "lex", "phi", "rin", "tha", "vor", "lin", "del"]

function syllablesFor(t: number): string[] {
  if (t < SYLLABLES.length) return SYLLABLES[t]
  // Build a stable pseudo-pool for higher type indices.
  const base = (t * 2654435761) >>> 0
  const a = FALLBACK_PRE[base % FALLBACK_PRE.length]
  const b = FALLBACK_POST[(base >>> 8) % FALLBACK_POST.length]
  return [a + b, b, a]
}

function mulberry32(seed: number) {
  let t = seed >>> 0
  return function () {
    t += 0x6d2b79f5
    let r = Math.imul(t ^ (t >>> 15), 1 | t)
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r)
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296
  }
}

/** Quantize histogram to 5% bins so tiny composition shifts don't shuffle the hash. */
export function hashHistogram(hist: number[]): number {
  let h = 2166136261 >>> 0
  for (let i = 0; i < hist.length; i++) {
    const q = Math.round(hist[i] * 20)
    h ^= (q << 8) ^ i
    h = Math.imul(h, 16777619) >>> 0
  }
  return h
}

type TypeShare = { type: number; share: number }

function topKTypes(hist: number[], k: number): TypeShare[] {
  const idx: TypeShare[] = []
  for (let t = 0; t < hist.length; t++) idx.push({ type: t, share: hist[t] })
  idx.sort((a, b) => b.share - a.share)
  return idx.slice(0, k)
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

/**
 * Birth-frozen colloquial name. Same composition produces the same name.
 * Returns short syllabic constructions like "Karonvi", "Drakthal", "Solmer".
 */
export function colloquialName(hist: number[]): string {
  const top = topKTypes(hist, 2).filter((t) => t.share > 0.05)
  if (top.length === 0) return "Voidlet"

  const seed = hashHistogram(hist)
  const rng = mulberry32(seed)

  const a = syllablesFor(top[0].type)
  const sa = a[Math.floor(rng() * a.length)]

  if (top.length === 1 || top[1].share < 0.15) {
    // monoculture — return one syllable plus a soft suffix
    const tail = ["en", "is", "ar", "yn", "or"][Math.floor(rng() * 5)]
    return capitalize(sa + tail)
  }

  const b = syllablesFor(top[1].type)
  const sb = b[Math.floor(rng() * b.length)]
  // avoid awkward double-letter joins (e.g. "lurell" → "lurell" is fine, but "kakal" is not)
  if (sa.endsWith(sb.charAt(0))) {
    return capitalize(sa + sb.slice(1))
  }
  return capitalize(sa + sb)
}

/**
 * Live engineering signature, e.g. `T0:62-T2:23#A3F1`.
 * Format: top-2 types with integer percentages, joined by `-`, then `#hash`.
 */
export function engineeringSignature(hist: number[]): string {
  const top = topKTypes(hist, 2).filter((t) => t.share > 0.05)
  if (top.length === 0) return "T?:??#0000"
  const parts = top.map((t) => `T${t.type}:${Math.round(t.share * 100)}`)
  const hash = (hashHistogram(hist) >>> 0)
    .toString(16)
    .toUpperCase()
    .padStart(8, "0")
    .slice(-4)
  return parts.join("-") + "#" + hash
}
