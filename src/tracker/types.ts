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
  /** Rolling samples of composite stability (same cadence/window as gates). */
  gateComposite: number[]
  /** Rolling samples of symmetry score. */
  gateSymmetry: number[]
}

export type Stability = {
  membership: number
  shape: number
  velocityCoherence: number
  size: number
  /** Bilateral symmetry about axes through CoM (~count balance in mirrored half-planes). */
  symmetry: number
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

  /**
   * Rolling mean of composite over `gateComposite` history (same window as tracker).
   * Used for leaderboard eligibility vs `stabilityGate`.
   */
  leaderboardAvgComposite: number
  /** Rolling mean of symmetry over `gateSymmetry` history. */
  leaderboardAvgSymmetry: number
  /**
   * True once rolling averages have met stability+symmetry gates at least once.
   * Grace-only removal applies after this (avoids grace before first qualify).
   */
  leaderboardEverMetGates: boolean
  /**
   * When rolling averages dipped below gates after `leaderboardEverMetGates`,
   * sim age (`ageSeconds`) at the start of that dip; used for grace period.
   */
  leaderboardGraceSinceAge: number | null
  /** Whether this organism counts for the competitors panel (average gates + grace). */
  leaderboardListed: boolean

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
  /** Max dead organisms retained. */
  deadCap: number
  /** Composite stability gate to enter visible top-N. */
  stabilityGate: number
  /** Symmetry gate ([0,1]): minimum bilateral symmetry score to appear in leaderboard. */
  symmetryGate: number
  /** Sim animation frames per tracker step (subsample to keep cost bounded). */
  framesPerStep: number
  /** Cap on visible alive list. */
  topN: number
  /**
   * After rolling averages dip below gates, keep showing the organism this many
   * sim seconds before dropping from the competitors list (unless averages recover).
   */
  leaderboardGraceSeconds: number
}

export const DEFAULT_TRACKER_PARAMS: TrackerParams = {
  bondFactor: 1.5,
  nMin: 4,
  tauMatch: 0.5,
  windowSize: 60,
  deadCap: 8,
  stabilityGate: 0.95,
  symmetryGate: 0.9,
  framesPerStep: 4,
  topN: 8,
  leaderboardGraceSeconds: 4,
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
  /**
   * Matches main canvas `layout.dpr` — use with `ImageData` width/height so
   * thumbnails display at one CSS pixel per device pixel (pixelated map match).
   */
  thumbnailDpr?: number
  /** `viewW / viewH` from main canvas — fixed-size thumb slots keep this aspect. */
  thumbnailViewAspect?: number
}
