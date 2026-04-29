/** Shared simulation types (CPU + WebGPU paths). */
export type Spec = {
  N: number
  K: number
  seed: number
  A: number[][]
  /** Per-pair minimum interaction radius (hard shell), K×K world units. */
  rMinMx: number[][]
  /** Per-pair maximum interaction radius (attraction cutoff), K×K. */
  RMx: number[][]
  /** Legacy scalars: defaults / migrating old saves. */
  rMin: number
  R: number
  dt: number
  /** Velocity damping: v *= max(0, 1 - friction). Matches Sandbox Science. */
  friction: number
  /** Hard-shell repulsion strength for distances below rMin. */
  repel: number
  /** Scales interaction forces before velocity integration. */
  forceFactor: number
  vMax: number
  wrap: boolean
  cellSize: number
  pixelScale: number
  genMatrix: boolean
  overlays: {
    showVel: boolean
    showGrid: boolean
    showTrails: boolean
    /** Debug: outline open-world trail FBO tiles (ignored when wrap is on). */
    showTrailTiles: boolean
  }
  /** Per frame, fraction of previous trail color kept (0.7–0.98). Higher = longer trails. */
  trailPersistence: number
  mutualOnly: boolean
  settleEnabled: boolean
  settleK: number
  settleR: number
  /**
   * How many fixed-dt physics steps to run per animation frame (1 = baseline).
   * Rounded and clamped to 1–50 in the UI. Higher = faster simulation time, more CPU/GPU load.
   */
  timeScale: number
}

export type Sim = {
  spec: Spec
  K: number
  x: Float32Array
  y: Float32Array
  vx: Float32Array
  vy: Float32Array
  type: Uint16Array
  interleaved: Float32Array
  fx: Float32Array
  fy: Float32Array
  /** Horizontal / vertical torus half-extents in world units (y fixed; x follows view aspect). */
  worldHalfW: number
  worldHalfH: number
  gridDimX: number
  gridDimY: number
  cellHead: Int32Array
  next: Int32Array
  rng: () => number
  frame: number
  lastMaxSpeed: number
  A: number[][]
  rMinMx: number[][]
  RMx: number[][]
  /** Open-world (!wrap): uniform-grid origin and bucket size (may be ≥ spec.cellSize if cloud is huge). */
  gridOriginX: number
  gridOriginY: number
  gridCellEff: number
}
