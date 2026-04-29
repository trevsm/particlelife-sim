/**
 * Organism tracker — public types.
 *
 * An organism is a set of particles that (1) is spatially connected via the
 * proximity graph, (2) persists across time under Jaccard-overlap matching, and
 * (3) accumulates coherence stats over a sliding window.
 */

export type DeathCause = "dissolution" | "split" | "absorbed"

/** A cluster detected in a single tracker step. Members are particle indices. */
export type Cluster = {
  members: Set<number>
  com: [number, number]
  vCom: [number, number]
  size: number
  /** Radius of gyration (RMS distance from CoM) in world units. */
  rg: number
  /** Normalized type histogram, length K. */
  typeHistogram: number[]
}

export type OrganismHistory = {
  com: [number, number][]
  size: number[]
  rg: number[]
  members: number[][]
}

export type Stability = {
  membership: number
  shape: number
  velocityCoherence: number
  size: number
  /** 0..1 weighted composite. */
  composite: number
}

export type Organism = {
  id: number
  /** Live engineering signature, e.g. "T0:62-T2:23#A3F1". Updates each step. */
  signature: string
  /** Birth-frozen colloquial name, e.g. "Karonvi". */
  name: string

  birthFrame: number
  lastSeenFrame: number
  deathFrame?: number
  deathCause?: DeathCause
  /** Surviving organism if this one was absorbed. */
  absorbedBy?: number
  /** Parent organism if this one was born from a split. */
  parentId?: number

  /** Current particle index set. */
  members: Set<number>
  com: [number, number]
  vCom: [number, number]
  size: number
  rg: number
  typeHistogram: number[]

  /** Cumulative path length of CoM in world units. */
  distance: number
  /** Approximate seconds alive (sim time, accounting for timeScale). */
  ageSeconds: number
  /** Tracker steps alive. */
  frames: number

  history: OrganismHistory
  stability: Stability
  speedAvg: number
  speedNow: number

  /** Composite score for ranking. */
  score: number

  thumbnail?: ImageData
}

export type TrackerParams = {
  /** Edge threshold = bondFactor × per-pair rMinMx. */
  bondFactor: number
  /** Min cluster size to count as an organism candidate. */
  nMin: number
  /** Jaccard threshold for matching across steps. */
  tauMatch: number
  /** Sliding window length in tracker steps. */
  windowSize: number
  /** Tracker steps between thumbnail refreshes. */
  snapshotInterval: number
  /** Max dead organisms retained. */
  deadCap: number
  /** Composite stability gate to enter visible top-N. */
  stabilityGate: number
  /** Sim animation frames per tracker step (subsample to keep cost bounded). */
  framesPerStep: number
  /** Cap on visible alive list. */
  topN: number
}

export const DEFAULT_TRACKER_PARAMS: TrackerParams = {
  bondFactor: 1.5,
  nMin: 4,
  tauMatch: 0.5,
  windowSize: 60,
  snapshotInterval: 30,
  deadCap: 8,
  stabilityGate: 0.55,
  framesPerStep: 4,
  topN: 8,
}

export type TrackerSnapshot = {
  /** Alive organisms sorted by score desc. */
  alive: Organism[]
  /** Recently dead, newest first, capped. */
  dead: Organism[]
  /** sim.frame at last tracker step. */
  frameAt: number
  /** Currently highlighted organism id (or null). */
  highlightId: number | null
}
