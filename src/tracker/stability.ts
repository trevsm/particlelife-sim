/**
 * Sliding-window stability + ranking math.
 *
 * - membership: how stable is the particle set across history (avg pairwise Jaccard)
 * - shape: how rigid is the cluster (variance of radius of gyration / mean rg)
 * - velocityCoherence: avg cosine between member velocity and CoM velocity
 * - size: stddev(N) / mean(N) inverted
 * - symmetry: bilateral balance across vertical/horizontal axes through CoM ([0,1])
 * - composite: weighted blend, all in [0,1]
 *
 * The composite score for ranking blends stability with longevity and size.
 * Logs on age and size keep one giant blob from drowning everyone else;
 * multiplicative on stability so unstable clusters can't grind out a high score
 * by living a long time.
 */

import type { Sim } from "../simTypes"
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

/** Bilateral symmetry: balance of particle counts across CoM-aligned half-planes, wrap-aware. */
export function symmetryScore(org: Organism, sim: Sim): number {
  const cx = org.com[0]
  const cy = org.com[1]
  const wrap = sim.spec.wrap
  const whw = sim.worldHalfW
  const whh = sim.worldHalfH
  const wW = 2 * whw
  const wH = 2 * whh

  let nLeft = 0
  let nRight = 0
  let nBelow = 0
  let nAbove = 0
  let n = 0
  for (const i of org.members) {
    let dx = sim.x[i] - cx
    let dy = sim.y[i] - cy
    if (wrap) {
      if (dx > whw) dx -= wW
      else if (dx < -whw) dx += wW
      if (dy > whh) dy -= wH
      else if (dy < -whh) dy += wH
    }
    n++
    if (dx <= 0) nLeft++
    else nRight++
    if (dy <= 0) nBelow++
    else nAbove++
  }

  if (n === 0) return 1
  const symX = 1 - Math.abs(nLeft - nRight) / n
  const symY = 1 - Math.abs(nBelow - nAbove) / n
  return Math.max(0, Math.min(1, (symX + symY) / 2))
}

export function computeStability(org: Organism, sim: Sim): Stability {
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

  const symmetry = symmetryScore(org, sim)

  const composite =
    0.31 * membership +
    0.21 * shape +
    0.21 * velocityCoherence +
    0.12 * size +
    0.15 * symmetry

  return {
    membership,
    shape,
    velocityCoherence,
    size,
    symmetry,
    composite,
  }
}

/** Composite score for the leaderboard. */
export function computeScore(org: Organism): number {
  const ageBonus = Math.log1p(org.ageSeconds)
  const sizeBonus = Math.log1p(org.size)
  return (
    ageBonus *
    (0.44 * org.stability.composite +
      0.26 * org.stability.velocityCoherence +
      0.15 * sizeBonus +
      0.15 * org.stability.symmetry)
  )
}

/** Push to a sliding-window array, trimming oldest entries past `cap`. */
export function pushCapped<T>(arr: T[], item: T, cap: number) {
  arr.push(item)
  if (arr.length > cap) arr.splice(0, arr.length - cap)
}
