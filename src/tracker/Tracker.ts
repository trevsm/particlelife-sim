/**
 * OrganismTracker — observes the sim each animation frame, runs cluster
 * detection on a subsampled cadence, matches clusters across steps via Jaccard,
 * accumulates per-organism stats, and produces snapshots for the UI.
 *
 * Lifecycle events:
 *   birth      — new cluster with no Jaccard match to any tracked organism
 *   continue   — cluster matched against an existing organism (Jaccard >= τ)
 *   dissolution — organism's members dispersed, no successor cluster
 *   split      — organism's members fragmented across two-or-more new clusters
 *   absorbed   — most of organism's members ended up inside a larger survivor
 *
 * Births that came from a parent organism's split keep `parentId` for lineage.
 * Absorbed organisms keep `absorbedBy` pointing at the survivor.
 */

import type { Sim } from "../simTypes"
import { detectClusters } from "./cluster"
import {
  computeScore,
  computeStability,
  jaccard,
  overlapFraction,
  pushCapped,
} from "./stability"
import {
  colloquialName,
  engineeringSignature,
} from "./naming"
import { renderThumbnail, type ThumbnailViewport } from "./thumbnail"
import {
  DEFAULT_TRACKER_PARAMS,
  type Cluster,
  type DeathCause,
  type Organism,
  type TrackerParams,
  type TrackerSnapshot,
} from "./types"

function meanSample(arr: number[]): number {
  if (arr.length === 0) return 0
  let s = 0
  for (let i = 0; i < arr.length; i++) s += arr[i]
  return s / arr.length
}

/** Live HUD thumbnails: target refresh rate vs animation frames (tracker step is slower). */
const HUD_THUMB_INTERVAL_MS = 1000 / 30

export class OrganismTracker {
  params: TrackerParams
  alive: Map<number, Organism> = new Map()
  dead: Organism[] = []

  private nextId = 1
  private trackerStep = 0
  private framesSinceStep = 0
  private lastSimFrame = 0
  /** Wall clock: throttle HUD thumbnail rasterization. */
  private lastHudThumbWallMs = 0
  private colors: string[]

  constructor(colors: string[], params: Partial<TrackerParams> = {}) {
    this.colors = colors
    this.params = { ...DEFAULT_TRACKER_PARAMS, ...params }
  }

  reset() {
    this.alive.clear()
    this.dead = []
    this.nextId = 1
    this.trackerStep = 0
    this.framesSinceStep = 0
    this.lastHudThumbWallMs = 0
  }

  /**
   * Call once per animation frame while the sim runs.
   * Refreshes live HUD thumbnails (~30fps) independently of tracker cluster steps.
   */
  observe(
    sim: Sim,
    thumbnailViewport?: ThumbnailViewport,
    opts?: { skipHudThumbnails?: boolean }
  ) {
    if (!opts?.skipHudThumbnails) {
      this.refreshHudThumbnails(sim, thumbnailViewport)
    }
    this.framesSinceStep++
    if (this.framesSinceStep < this.params.framesPerStep) return
    this.framesSinceStep = 0
    this.step(sim)
  }

  /** Top-N listed competitors: live view at ~30fps using current particle positions. */
  private refreshHudThumbnails(sim: Sim, vp?: ThumbnailViewport) {
    if (!vp || this.alive.size === 0) return
    const now =
      typeof performance !== "undefined" ? performance.now() : 0
    if (now - this.lastHudThumbWallMs < HUD_THUMB_INTERVAL_MS) return
    this.lastHudThumbWallMs = now

    const list = Array.from(this.alive.values())
      .filter((o) => o.leaderboardListed)
      .sort((a, b) => b.score - a.score)
      .slice(0, this.params.topN)

    for (const org of list) {
      try {
        org.thumbnail = renderThumbnail(org, sim, this.colors, vp)
      } catch {
        // Canvas creation can fail in some headless envs; non-fatal.
      }
    }
  }

  private step(sim: Sim) {
    this.trackerStep++
    this.lastSimFrame = sim.frame

    const dtSecPerStep = sim.spec.dt * this.params.framesPerStep
    const clusters = detectClusters(sim, this.params)

    // Build greedy bipartite match: best Jaccard wins.
    const aliveList = Array.from(this.alive.values())
    type Pair = { orgId: number; clusterIdx: number; jacc: number }
    const candidates: Pair[] = []
    for (const org of aliveList) {
      for (let ci = 0; ci < clusters.length; ci++) {
        const j = jaccard(org.members, clusters[ci].members)
        if (j > 0) candidates.push({ orgId: org.id, clusterIdx: ci, jacc: j })
      }
    }
    candidates.sort((a, b) => b.jacc - a.jacc)

    const orgsTaken = new Set<number>()
    const clustersTaken = new Set<number>()
    const assignments: Pair[] = []
    for (const p of candidates) {
      if (orgsTaken.has(p.orgId) || clustersTaken.has(p.clusterIdx)) continue
      if (p.jacc < this.params.tauMatch) continue
      orgsTaken.add(p.orgId)
      clustersTaken.add(p.clusterIdx)
      assignments.push(p)
    }

    // Apply continuations.
    for (const a of assignments) {
      const org = this.alive.get(a.orgId)
      if (!org) continue
      this.updateOrganism(org, clusters[a.clusterIdx], sim, dtSecPerStep)
    }

    // Determine deaths and their causes BEFORE births, since absorbed-into-survivor
    // detection looks at organisms still in `alive`.
    const dyingIds: number[] = []
    for (const org of aliveList) {
      if (orgsTaken.has(org.id)) continue
      dyingIds.push(org.id)
    }

    // For each dying org, classify cause based on overlap with new clusters and survivors.
    const dyingClassified: { org: Organism; cause: DeathCause; absorbedBy?: number; splitChildren?: number[] }[] = []
    for (const id of dyingIds) {
      const org = this.alive.get(id)!

      // Look for a survivor (existing alive that took matched cluster) whose new
      // member set absorbed >60% of the dying org's members.
      let absorbedBy: number | undefined
      let bestAbsorb = 0
      for (const a of assignments) {
        const survivor = this.alive.get(a.orgId)
        if (!survivor || survivor.id === org.id) continue
        const ratio = overlapFraction(org.members, survivor.members)
        if (ratio > 0.6 && ratio > bestAbsorb) {
          bestAbsorb = ratio
          absorbedBy = survivor.id
        }
      }

      // Split: 2+ new clusters each contain >25% of the dying org's old members.
      const splitChildren: number[] = []
      if (absorbedBy === undefined) {
        for (let ci = 0; ci < clusters.length; ci++) {
          if (clustersTaken.has(ci)) continue // already became a continuation
          const ratio = overlapFraction(org.members, clusters[ci].members)
          if (ratio > 0.25) splitChildren.push(ci)
        }
      }

      let cause: DeathCause = "dissolution"
      if (absorbedBy !== undefined) cause = "absorbed"
      else if (splitChildren.length >= 2) cause = "split"

      dyingClassified.push({
        org,
        cause,
        absorbedBy,
        splitChildren: splitChildren.length >= 2 ? splitChildren : undefined,
      })
    }

    // Births: clusters not matched in assignments. If a birth came from a split,
    // tag parent.
    const splitParentByCluster = new Map<number, number>()
    for (const d of dyingClassified) {
      if (d.cause !== "split" || !d.splitChildren) continue
      for (const ci of d.splitChildren) splitParentByCluster.set(ci, d.org.id)
    }

    for (let ci = 0; ci < clusters.length; ci++) {
      if (clustersTaken.has(ci)) continue
      this.createOrganism(clusters[ci], sim, splitParentByCluster.get(ci))
    }

    // Now apply deaths.
    for (const d of dyingClassified) {
      d.org.deathFrame = sim.frame
      d.org.deathCause = d.cause
      if (d.absorbedBy !== undefined) d.org.absorbedBy = d.absorbedBy
      this.alive.delete(d.org.id)
      this.dead.unshift(d.org)
    }
    if (this.dead.length > this.params.deadCap) {
      this.dead.length = this.params.deadCap
    }
  }

  private updateOrganism(
    org: Organism,
    c: Cluster,
    sim: Sim,
    dtSecPerStep: number
  ) {
    const wrap = sim.spec.wrap
    const whw = sim.worldHalfW
    const whh = sim.worldHalfH

    // Distance increment, accounting for wrap.
    let dx = c.com[0] - org.com[0]
    let dy = c.com[1] - org.com[1]
    if (wrap) {
      const wW = 2 * whw
      const wH = 2 * whh
      if (dx > whw) dx -= wW
      else if (dx < -whw) dx += wW
      if (dy > whh) dy -= wH
      else if (dy < -whh) dy += wH
    }
    const stepDist = Math.sqrt(dx * dx + dy * dy)

    org.members = c.members
    org.com = c.com
    org.vCom = c.vCom
    org.size = c.size
    org.rg = c.rg
    org.typeHistogram = c.typeHistogram
    org.signature = engineeringSignature(c.typeHistogram)
    org.lastSeenFrame = sim.frame
    org.frames++
    org.distance += stepDist
    org.ageSeconds += dtSecPerStep

    org.speedNow = Math.sqrt(c.vCom[0] * c.vCom[0] + c.vCom[1] * c.vCom[1])
    org.speedAvg = org.speedAvg * 0.85 + org.speedNow * 0.15

    pushCapped(org.history.com, [c.com[0], c.com[1]], this.params.windowSize)
    pushCapped(org.history.size, c.size, this.params.windowSize)
    pushCapped(org.history.rg, c.rg, this.params.windowSize)
    pushCapped(org.history.members, Array.from(c.members), this.params.windowSize)

    org.stability = computeStability(org, sim)
    org.score = computeScore(org)

    pushCapped(
      org.history.gateComposite,
      org.stability.composite,
      this.params.windowSize
    )
    pushCapped(
      org.history.gateSymmetry,
      org.stability.symmetry,
      this.params.windowSize
    )

    const avgC =
      org.history.gateComposite.length > 0
        ? meanSample(org.history.gateComposite)
        : org.stability.composite
    const avgS =
      org.history.gateSymmetry.length > 0
        ? meanSample(org.history.gateSymmetry)
        : org.stability.symmetry
    org.leaderboardAvgComposite = avgC
    org.leaderboardAvgSymmetry = avgS

    const stabGate = this.params.stabilityGate
    const symGate = this.params.symmetryGate
    const avgPass = avgC >= stabGate && avgS >= symGate
    const grace = this.params.leaderboardGraceSeconds

    if (avgPass) {
      org.leaderboardEverMetGates = true
      org.leaderboardGraceSinceAge = null
      org.leaderboardListed = true
    } else if (org.leaderboardEverMetGates) {
      if (org.leaderboardGraceSinceAge === null) {
        org.leaderboardGraceSinceAge = org.ageSeconds
      }
      const inGrace =
        org.ageSeconds - org.leaderboardGraceSinceAge < grace
      org.leaderboardListed = inGrace
    } else {
      org.leaderboardGraceSinceAge = null
      org.leaderboardListed = false
    }
  }

  private createOrganism(c: Cluster, sim: Sim, parentId?: number) {
    const id = this.nextId++
    const speedNow = Math.sqrt(c.vCom[0] * c.vCom[0] + c.vCom[1] * c.vCom[1])
    const org: Organism = {
      id,
      signature: engineeringSignature(c.typeHistogram),
      name: colloquialName(c.typeHistogram),
      birthFrame: sim.frame,
      lastSeenFrame: sim.frame,
      members: c.members,
      com: c.com,
      vCom: c.vCom,
      size: c.size,
      rg: c.rg,
      typeHistogram: c.typeHistogram,
      distance: 0,
      ageSeconds: 0,
      frames: 1,
      history: {
        com: [[c.com[0], c.com[1]]],
        size: [c.size],
        rg: [c.rg],
        members: [Array.from(c.members)],
        gateComposite: [],
        gateSymmetry: [],
      },
      stability: {
        membership: 1,
        shape: 1,
        velocityCoherence: 0.5,
        size: 1,
        symmetry: 1,
        composite: 0.895,
      },
      speedAvg: speedNow,
      speedNow,
      score: 0,
      leaderboardAvgComposite: 0.895,
      leaderboardAvgSymmetry: 1,
      leaderboardEverMetGates: false,
      leaderboardGraceSinceAge: null,
      leaderboardListed: false,
    }
    if (parentId !== undefined) org.parentId = parentId
    org.score = computeScore(org)
    this.alive.set(id, org)
  }

  /** Build a snapshot for the UI. Cheap: O(alive). */
  snapshot(highlightId: number | null = null): TrackerSnapshot {
    const alive = Array.from(this.alive.values())
    alive.sort((a, b) => b.score - a.score)
    return {
      alive,
      dead: this.dead.slice(),
      frameAt: this.lastSimFrame,
      highlightId,
    }
  }
}
