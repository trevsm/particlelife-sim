/**
 * Sliding-window stability + ranking math.
 *
 * - membership: how stable is the particle set across history (avg pairwise Jaccard)
 * - shape: how rigid is the cluster (variance of radius of gyration / mean rg)
 * - velocityCoherence: avg cosine between member velocity and CoM velocity
 * - size: stddev(N) / mean(N) inverted
 * - composite: weighted blend, all in [0,1]
 *
 * The composite score for ranking blends stability with longevity and size.
 * Logs on age and size keep one giant blob from drowning everyone else;
 * multiplicative on stability so unstable clusters can't grind out a high score
 * by living a long time.
 */

import type { Organism, Stability } from "./types"

export function jaccard(a: Set<number>, b: Set<number>): number {
  if (a.size === 0 && b.size === 0) return 1
  let inter = 0
  // iterate the smaller set
  if (a.size <= b.size) {
    for (const x of a) if (b.has(x)) inter++
  } else {
    for (const x of b) if (a.has(x)) inter++
  }
  const union = a.size + b.size - inter
  if (union === 0) return 0
  return inter / union
}

/**
 * Compute member-set overlap as a fraction of the smaller set.
 * Used for split/absorb detection: "did most of A end up in B?"
 */
export function overlapFraction(a: Set<number>, b: Set<number>): number {
  if (a.size === 0) return 0
  let inter = 0
  if (a.size <= b.size) {
    for (const x of a) if (b.has(x)) inter++
  } else {
    for (const x of b) if (a.has(x)) inter++
  }
  return inter / a.size
}

type VelView = { vx: Float32Array; vy: Float32Array }

export function computeStability(org: Organism, sim: VelView): Stability {
  const hist = org.history

  // membership stability — avg Jaccard between consecutive history snapshots
  let membership = 1
  if (hist.members.length >= 2) {
    let sum = 0
    let n = 0
    let prev = new Set(hist.members[0])
    for (let i = 1; i < hist.members.length; i++) {
      const curr = new Set(hist.members[i])
      sum += jaccard(prev, curr)
      n++
      prev = curr
    }
    membership = n > 0 ? sum / n : 1
  }

  // shape stability via radius of gyration variance / mean
  let shape = 1
  if (hist.rg.length >= 2) {
    let sum = 0
    for (const v of hist.rg) sum += v
    const mean = sum / hist.rg.length
    if (mean > 1e-9) {
      let s2 = 0
      for (const v of hist.rg) {
        const d = v - mean
        s2 += d * d
      }
      const std = Math.sqrt(s2 / hist.rg.length)
      shape = Math.max(0, 1 - std / mean)
    }
  }

  // size stability
  let size = 1
  if (hist.size.length >= 2) {
    let sum = 0
    for (const v of hist.size) sum += v
    const mean = sum / hist.size.length
    if (mean > 1e-9) {
      let s2 = 0
      for (const v of hist.size) {
        const d = v - mean
        s2 += d * d
      }
      const std = Math.sqrt(s2 / hist.size.length)
      size = Math.max(0, 1 - std / mean)
    }
  }

  // velocity coherence — avg cos(v_i, v_com)
  let velocityCoherence: number
  const vcx = org.vCom[0]
  const vcy = org.vCom[1]
  const vcLen = Math.sqrt(vcx * vcx + vcy * vcy)
  if (vcLen > 1e-9) {
    let sum = 0
    let n = 0
    for (const i of org.members) {
      const vix = sim.vx[i]
      const viy = sim.vy[i]
      const viLen = Math.sqrt(vix * vix + viy * viy)
      if (viLen > 1e-9) {
        sum += (vix * vcx + viy * vcy) / (viLen * vcLen)
        n++
      }
    }
    velocityCoherence = n > 0 ? Math.max(0, sum / n) : 0
  } else {
    // CoM is near-stationary. If individual particles are too, treat as coherent rest;
    // if they're moving, treat as low coherence.
    let energy = 0
    let n = 0
    for (const i of org.members) {
      energy += sim.vx[i] * sim.vx[i] + sim.vy[i] * sim.vy[i]
      n++
    }
    const meanSpeed = n > 0 ? Math.sqrt(energy / n) : 0
    velocityCoherence = meanSpeed < 0.05 ? 0.7 : 0.2
  }

  const composite =
    0.35 * membership +
    0.25 * shape +
    0.25 * velocityCoherence +
    0.15 * size

  return {
    membership,
    shape,
    velocityCoherence,
    size,
    composite,
  }
}

/** Composite score for the leaderboard. */
export function computeScore(org: Organism): number {
  const ageBonus = Math.log1p(org.ageSeconds)
  const sizeBonus = Math.log1p(org.size)
  return (
    ageBonus *
    (0.5 * org.stability.composite +
      0.3 * org.stability.velocityCoherence +
      0.2 * sizeBonus)
  )
}

/** Push to a sliding-window array, trimming oldest entries past `cap`. */
export function pushCapped<T>(arr: T[], item: T, cap: number) {
  arr.push(item)
  if (arr.length > cap) arr.splice(0, arr.length - cap)
}
