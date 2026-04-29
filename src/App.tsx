import React, { useEffect, useRef, useState } from "react"
import type { Sim, Spec } from "./simTypes"
import {
  createParticleGL,
  drawParticles,
  ensureParticleBufferCapacity,
  resizeDrawingSurface,
  type ParticleGL,
  type ViewLayout,
} from "./webglParticles"
import { createGpuSimRunner, type GpuSimRunner } from "./webgpuSim"

/**
 * Particle Life backend + editable ruleset (A matrix) with toolbar.
 * - World in [-1,1]^2; optional wrap.
 * - Pairwise accelerator with rMin (Particle-Life style).
 * - Second-order dynamics with friction (velocity damping) and vMax clamp.
 * - Uniform-grid neighbor search.
 * - Toolbar shows a K×K clickable grid to edit A.
 *   * Left-click cycles value: -1 → 0 → +1 → …
 *   * Right-click cycles the other direction.
 *   * Values are saved to localStorage and applied live.
 * - “Ring preset”: each type attracts itself and its next color (i→i, i→i+1).
 *
 * Rendering:
 * - WebGL2 point sprites for particles (default N up to 1M draw in one draw call).
 * - Transparent 2D canvas overlay for grid, HUD, velocity vectors (velocity subsampled when N is huge).
 * - When WebGPU is available, **forces + integration** run on the GPU (atomic
 *   accumulation; same rules as CPU). The spatial grid is built on the CPU and
 *   uploaded each frame; positions are read back for WebGL drawing.
 *
 * Notes on stability:
 * - Re-initialize simulation buffers whenever core layout changes (N, K, cellSize, etc.).
 * - Never loop past typed-array lengths; use safe N.
 * - Do not early-return from re-init when A exists in localStorage.
 *
 * Non-reciprocal update:
 * - Interactions are now non-reciprocal: F_ij and F_ji are computed from A[i][j] and A[j][i] separately.
 *   This allows self-propelled clusters when A is asymmetric.
 */

// ========================= Spec =========================
const TYPE_COLORS = [
  "#FF5A5A", // red
  "#FF8A3C", // orange
  "#FFD23B", // yellow
  "#7DDE3B", // yellow-green
  "#34C759", // green
  "#30D7A9", // aquamarine
  "#4AA8FF", // blue
  "#6E6BFF", // indigo
  "#A15BFF", // violet
  "#FF5BD1", // magenta
  "#FF7F50", // coral
  "#00CED1", // dark turquoise
  "#708090", // slate gray
  "#7FFF00", // chartreuse
  "#CC79A7", // pink (okabe-ito)
  "#ffffff", // white
]

/** Max particles (WebGL draws in one call; physics is still CPU spatial-hash). */
const MAX_PARTICLES = 1_000_000
const ENABLE_WEBGPU_PHYSICS = true

// ========================= RNG =========================
function mulberry32(seed: number) {
  let t = seed >>> 0
  return function () {
    t += 0x6d2b79f5
    let r = Math.imul(t ^ (t >>> 15), 1 | t)
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r)
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296
  }
}
function randRange(rng: () => number, a: number, b: number) {
  return a + (b - a) * rng()
}
function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v))
}

function cloneMx(m: number[][]): number[][] {
  return m.map((row) => row.slice())
}

function maxRMx(RMx: number[][]): number {
  let m = 0.1
  for (const row of RMx) {
    for (const v of row) {
      if (Number.isFinite(v) && v > m) m = v
    }
  }
  return m
}

/** Per-pair radii that pair with ring force preset. */
function genRingRadiusPreset(K: number, r0: number, R0: number) {
  const rMinMx: number[][] = Array.from({ length: K }, () => Array(K).fill(r0 * 0.72))
  const RMx: number[][] = Array.from({ length: K }, () => Array(K).fill(R0 * 0.88))
  for (let i = 0; i < K; i++) {
    rMinMx[i][i] = r0 * 1.05
    rMinMx[i][(i + 1) % K] = r0 * 0.92
    RMx[i][i] = R0 * 1.08
    RMx[i][(i + 1) % K] = R0 * 1.02
  }
  return { rMinMx, RMx }
}

// ========================= World helpers =========================
const WORLD_SIZE = 2.0 // [-1,1]

// ========================= Particle Life accelerator =========================
/** Sandbox Science triangular force curve; hard repulsion inside rMin. */
function accelMag(
  a: number,
  r: number,
  rMinP: number,
  RP: number,
  repel: number
): number {
  if (r <= 0) return 0
  if (r < rMinP) return (r / rMinP) * repel - repel
  if (r > RP) return 0
  const denom = Math.max(1e-6, RP - rMinP)
  return a * (1 - Math.abs(rMinP + RP - 2 * r) / denom)
}

// ========================= A matrix helpers =========================
function genRandomMatrix(K: number, rng: () => number) {
  const A: number[][] = Array.from({ length: K }, () => Array(K).fill(0))
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) {
      A[i][j] = i === j ? randRange(rng, 0.2, 0.5) : randRange(rng, -0.45, 0.45)
    }
  }
  return A
}

const DEFAULT_R_MIN = 0.035
const DEFAULT_R_MAX = 0.1
const DEFAULT_SELF_FORCE = 0.42
const DEFAULT_NEXT_FORCE = 0.26
const DEFAULT_OTHER_FORCE = -0.04
const DEFAULT_REPEL = 0.46
const DEFAULT_FORCE_FACTOR = 1

/** Ring preset: each type attracts itself and its next color (i→i, i→i+1). */
function genRingPreset(
  K: number,
  self = DEFAULT_SELF_FORCE,
  next = DEFAULT_NEXT_FORCE,
  others = DEFAULT_OTHER_FORCE
) {
  const A: number[][] = Array.from({ length: K }, () => Array(K).fill(others))
  for (let i = 0; i < K; i++) {
    A[i][i] = self
    A[i][(i + 1) % K] = next
  }
  return A
}

const DEFAULT_K = 7

/** Reference UI matrices (7×7), matched to the default Vue particle-life controls. */
const REF_DEFAULT_A: number[][] = [
  [0.34, -0.92, -0.13, 0.22, -0.25, -0.17, -0.74],
  [-0.1, -0.26, -0.19, 0.58, 0.64, -0.75, 0.59],
  [-0.12, -0.55, 0.44, 0.72, 0.44, -0.95, -0.22],
  [0.84, -0.24, 0.84, -0.82, 0.65, 0.46, -0.7],
  [-0.25, 0.37, 0.7, 0.21, 0.45, -0.27, -0.71],
  [0.77, -0.64, -0.27, -0.42, 0.84, -0.12, 0.61],
  [0.94, -0.17, 0.13, 0.8, -0.5, -0.58, -0.36],
]

const REF_WORLD_UNITS_PER_PX = WORLD_SIZE / 1280

/** Slider tick labels from that UI (12–24 px); mapped into our [-1,1] world. */
const REF_RMIN_UI: number[][] = [
  [21, 24, 13, 17, 15, 16, 17],
  [20, 14, 12, 16, 19, 16, 22],
  [15, 23, 15, 20, 16, 21, 16],
  [23, 19, 22, 13, 24, 19, 14],
  [12, 13, 13, 12, 12, 12, 23],
  [23, 17, 12, 20, 12, 17, 21],
  [21, 22, 17, 18, 22, 13, 14],
]

/** Slider tick labels (33–63 px); mapped into our [-1,1] world. */
const REF_RMAX_UI: number[][] = [
  [52, 58, 36, 58, 44, 44, 33],
  [39, 55, 63, 58, 63, 43, 54],
  [34, 63, 52, 42, 44, 60, 37],
  [42, 60, 61, 49, 63, 36, 48],
  [63, 38, 59, 39, 38, 40, 46],
  [49, 54, 54, 42, 44, 63, 54],
  [62, 42, 62, 36, 53, 41, 48],
]

function refUiMinRToWorld(v: number): number {
  return clamp(v, 12, 24) * REF_WORLD_UNITS_PER_PX
}

function refUiMaxRToWorld(v: number): number {
  return clamp(v, 33, 63) * REF_WORLD_UNITS_PER_PX
}

function buildRefDefaultRadii(): { rMinMx: number[][]; RMx: number[][] } {
  const K = REF_RMIN_UI.length
  const pairEps = 0.002
  const rMinMx: number[][] = Array.from({ length: K }, () => [])
  const RMx: number[][] = Array.from({ length: K }, () => [])
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) {
      const rmin = refUiMinRToWorld(REF_RMIN_UI[i][j])
      const rmax = Math.max(
        refUiMaxRToWorld(REF_RMAX_UI[i][j]),
        rmin + pairEps
      )
      rMinMx[i][j] = rmin
      RMx[i][j] = rmax
    }
  }
  return { rMinMx, RMx }
}

const SPEC_REF_R = buildRefDefaultRadii()

/** Default spec: reference Vue matrices + high damping for stable clusters. */
const SPEC: Spec = {
  N: 6_000,
  K: DEFAULT_K,
  seed: 1337,

  A: cloneMx(REF_DEFAULT_A),
  rMinMx: cloneMx(SPEC_REF_R.rMinMx),
  RMx: cloneMx(SPEC_REF_R.RMx),
  rMin: DEFAULT_R_MIN,
  R: DEFAULT_R_MAX,

  dt: 1 / 60,
  friction: 0.3,
  repel: DEFAULT_REPEL,
  forceFactor: DEFAULT_FORCE_FACTOR,
  vMax: 4,

  wrap: true,
  cellSize: 0.05,

  pixelScale: 800,
  genMatrix: false,

  overlays: { showVel: false, showGrid: false },

  mutualOnly: false,
  settleEnabled: false,
  settleK: 0.08,
  settleR: 0.08,
}

const LS_KEY_V2 = "pl_rules_v5"
const LS_KEY_V1 = "pl_rules_v1"

function saveRulesToLS(
  K: number,
  A: number[][],
  rMinMx: number[][],
  RMx: number[][]
) {
  try {
    localStorage.setItem(LS_KEY_V2, JSON.stringify({ K, A, rMinMx, RMx }))
  } catch {}
}

function loadRulesFromLS(
  K: number,
  fallbackRMin: number,
  fallbackR: number
): { A: number[][]; rMinMx: number[][]; RMx: number[][] } | null {
  try {
    void fallbackRMin
    void fallbackR
    const raw2 = localStorage.getItem(LS_KEY_V2)
    if (raw2) {
      const obj = JSON.parse(raw2)
      if (obj && obj.K === K && obj.A && obj.rMinMx && obj.RMx) {
        const { A, rMinMx, RMx } = obj
        if (
          Array.isArray(A) &&
          A.length === K &&
          !A.some((r: unknown) => !Array.isArray(r) || r.length !== K) &&
          Array.isArray(rMinMx) &&
          rMinMx.length === K &&
          !rMinMx.some((r: unknown) => !Array.isArray(r) || r.length !== K) &&
          Array.isArray(RMx) &&
          RMx.length === K &&
          !RMx.some((r: unknown) => !Array.isArray(r) || r.length !== K)
        ) {
          return {
            A: A as number[][],
            rMinMx: rMinMx as number[][],
            RMx: RMx as number[][],
          }
        }
      }
    }
    void LS_KEY_V1
    return null
  } catch {
    return null
  }
}

// ========================= Simulation state =========================
function initSim(spec: Spec, seedOverride?: number): Sim {
  const rng = mulberry32(seedOverride ?? spec.seed)
  const K = spec.K

  // choose rules: localStorage → preset (ring) → spec
  const loaded = loadRulesFromLS(K, spec.rMin, spec.R)
  let A: number[][]
  let rMinMx: number[][]
  let RMx: number[][]
  if (loaded) {
    A = loaded.A
    const sane = sanitizeRadiusMatrices(loaded.rMinMx, loaded.RMx)
    rMinMx = sane.rMinMx
    RMx = sane.RMx
  } else if (spec.genMatrix) {
    A = genRingPreset(K)
    const ring = genRingRadiusPreset(K, spec.rMin, spec.R)
    rMinMx = ring.rMinMx
    RMx = ring.RMx
  } else {
    A = spec.A
    const ring = genRingRadiusPreset(K, spec.rMin, spec.R)
    rMinMx = spec.rMinMx.length === K ? cloneMx(spec.rMinMx) : ring.rMinMx
    RMx = spec.RMx.length === K ? cloneMx(spec.RMx) : ring.RMx
  }

  const x = new Float32Array(spec.N)
  const y = new Float32Array(spec.N)
  const vx = new Float32Array(spec.N)
  const vy = new Float32Array(spec.N)
  const type = new Uint16Array(spec.N)
  const interleaved = new Float32Array(spec.N * 3)

  for (let i = 0; i < spec.N; i++) {
    x[i] = randRange(rng, -1, 1)
    y[i] = randRange(rng, -1, 1)
    vx[i] = randRange(rng, -0.005, 0.005)
    vy[i] = randRange(rng, -0.005, 0.005)
    type[i] = Math.floor(rng() * K)
  }

  const gridDim = Math.max(1, Math.ceil(WORLD_SIZE / spec.cellSize))
  const cellHead = new Int32Array(gridDim * gridDim)
  const next = new Int32Array(spec.N)

  return {
    spec,
    K,
    x,
    y,
    vx,
    vy,
    type,
    interleaved,
    fx: new Float32Array(spec.N),
    fy: new Float32Array(spec.N),
    gridDim,
    cellHead,
    next,
    rng,
    frame: 0,
    lastMaxSpeed: 0,
    A: cloneMx(A),
    rMinMx: cloneMx(rMinMx),
    RMx: cloneMx(RMx),
  }
}

// ========================= Neighbors =========================
function cellIndexOf(x: number, y: number, gridDim: number): number {
  const gx = clamp(Math.floor((x + 1) * 0.5 * gridDim), 0, gridDim - 1)
  const gy = clamp(Math.floor((y + 1) * 0.5 * gridDim), 0, gridDim - 1)
  return gx + gy * gridDim
}
function rebuildGrid(sim: Sim) {
  sim.cellHead.fill(-1)
  // Safe N prevents writing past typed-array bounds after spec changes.
  const N = Math.min(sim.spec.N, sim.x.length, sim.next.length)
  const gdim = sim.gridDim
  for (let i = 0; i < N; i++) {
    const idx = cellIndexOf(sim.x[i], sim.y[i], gdim)
    sim.next[i] = sim.cellHead[idx]
    sim.cellHead[idx] = i
  }
}
// ========================= Physics step =========================
function smoothstep01(t: number) {
  const x = clamp(t, 0, 1)
  return x * x * (3 - 2 * x)
}

function step(sim: Sim) {
  const sp = sim.spec
  // Safe N: never iterate beyond buffer lengths.
  const N = Math.min(sp.N, sim.x.length, sim.vx.length, sim.type.length)
  const dt = sp.dt

  rebuildGrid(sim)

  // Hoist hot fields as monomorphic locals (avoids repeated property lookup).
  const X = sim.x
  const Y = sim.y
  const VX = sim.vx
  const VY = sim.vy
  const FX = sim.fx
  const FY = sim.fy
  const TYPE = sim.type
  const A = sim.A
  const RMINMX = sim.rMinMx
  const RMAXMX = sim.RMx
  const cellHead = sim.cellHead
  const linkNext = sim.next
  const gdim = sim.gridDim
  const wrap = sp.wrap
  const cellSize = sp.cellSize
  const mutualOnly = sp.mutualOnly
  const repel = sp.repel
  const forceFactor = sp.forceFactor
  const settleEnabled = sp.settleEnabled
  const settleR = sp.settleR
  const settleK = sp.settleK
  const cCrit = 2 / dt
  const cSettle = Math.min(Math.max(settleK, 0), cCrit)
  const invSettleSpan = 1 / Math.max(1e-6, settleR - 0)

  // Cache the per-frame neighbor reach (was being recomputed per-particle).
  const maxR = maxRMx(RMAXMX)
  const maxR2 = maxR * maxR
  const reach = Math.max(1, Math.ceil(maxR / cellSize))
  const halfG = 0.5 * gdim

  FX.fill(0, 0, N)
  FY.fill(0, 0, N)

  for (let i = 0; i < N; i++) {
    const ti = TYPE[i] | 0
    const xi = X[i]
    const yi = Y[i]
    const ARi = A[ti]
    const RminRi = RMINMX[ti]
    const RmaxRi = RMAXMX[ti]

    const cxi = (xi + 1) * halfG
    const cyi = (yi + 1) * halfG
    const cx = cxi < 0 ? 0 : cxi >= gdim ? gdim - 1 : cxi | 0
    const cy = cyi < 0 ? 0 : cyi >= gdim ? gdim - 1 : cyi | 0

    let lfx = 0
    let lfy = 0

    for (let dy = -reach; dy <= reach; dy++) {
      let ny = cy + dy
      if (wrap) {
        ny = ((ny % gdim) + gdim) % gdim
      } else if (ny < 0 || ny >= gdim) {
        continue
      }
      const rowBase = ny * gdim
      for (let dx = -reach; dx <= reach; dx++) {
        let nx = cx + dx
        if (wrap) {
          nx = ((nx % gdim) + gdim) % gdim
        } else if (nx < 0 || nx >= gdim) {
          continue
        }

        let j = cellHead[nx + rowBase]
        while (j !== -1) {
          if (j > i) {
            const xj = X[j]
            const yj = Y[j]
            let ddx = xj - xi
            let ddy = yj - yi
            if (wrap) {
              if (ddx > 1) ddx -= WORLD_SIZE
              else if (ddx < -1) ddx += WORLD_SIZE
              if (ddy > 1) ddy -= WORLD_SIZE
              else if (ddy < -1) ddy += WORLD_SIZE
            }
            const r2 = ddx * ddx + ddy * ddy
            if (r2 > 0 && r2 <= maxR2) {
              const tj = TYPE[j] | 0
              const R_ij = RmaxRi[tj]
              const R_ji = RMAXMX[tj][ti]
              if (r2 <= R_ij * R_ij || r2 <= R_ji * R_ji) {
                const r = Math.sqrt(r2)
                const invr = 1 / r
                const ux = ddx * invr
                const uy = ddy * invr

                let aij = ARi[tj]
                let aji = A[tj][ti]
                if (mutualOnly) {
                  if (aij <= 0 || aji <= 0) {
                    aij = 0
                    aji = 0
                  }
                }

                const r_ij = RminRi[tj]
                const r_ji = RMINMX[tj][ti]
                const fij = r <= R_ij ? accelMag(aij, r, r_ij, R_ij, repel) : 0
                const fji = r <= R_ji ? accelMag(aji, r, r_ji, R_ji, repel) : 0

                if (fij !== 0) {
                  lfx += fij * ux
                  lfy += fij * uy
                }
                if (fji !== 0) {
                  FX[j] -= fji * ux
                  FY[j] -= fji * uy
                }

                if (settleEnabled && aij > 0 && aji > 0) {
                  const rShell = r_ij < r_ji ? r_ij : r_ji
                  if (r > rShell && r < settleR) {
                    const vRelRad =
                      (VX[i] - VX[j]) * ux + (VY[i] - VY[j]) * uy
                    const span = settleR - rShell
                    if (span > 1e-6) {
                      const t = (r - rShell) / span
                      const w = 1 - smoothstep01(t)
                      if (w > 0) {
                        const fDamp = -cSettle * vRelRad * w
                        const sfx = fDamp * ux
                        const sfy = fDamp * uy
                        lfx += sfx
                        lfy += sfy
                        FX[j] -= sfx
                        FY[j] -= sfy
                      }
                    }
                  }
                }
              }
            }
          }
          j = linkNext[j]
        }
      }
    }

    FX[i] += lfx
    FY[i] += lfy
  }
  // Suppress lint warnings for cached scalar that is intentionally unused.
  void invSettleSpan

  // integrate
  let maxSpeed2 = 0
  const frictionFactor = Math.max(0, 1 - sp.friction)
  const vMax = sp.vMax
  const vMax2 = vMax * vMax
  const scaledForceFactor = forceFactor * REF_WORLD_UNITS_PER_PX

  for (let i = 0; i < N; i++) {
    let vxi = (VX[i] + FX[i] * scaledForceFactor) * frictionFactor
    let vyi = (VY[i] + FY[i] * scaledForceFactor) * frictionFactor

    let v2 = vxi * vxi + vyi * vyi
    if (v2 > vMax2) {
      const s = vMax / Math.sqrt(v2)
      vxi *= s
      vyi *= s
      v2 = vMax2
    }
    if (v2 > maxSpeed2) maxSpeed2 = v2

    let nx = X[i] + dt * vxi
    let ny = Y[i] + dt * vyi
    if (wrap) {
      if (nx < -1) nx += WORLD_SIZE
      else if (nx > 1) nx -= WORLD_SIZE
      if (ny < -1) ny += WORLD_SIZE
      else if (ny > 1) ny -= WORLD_SIZE
    } else {
      if (nx < -1) {
        nx = -1 + (-1 - nx)
        vxi = Math.abs(vxi)
      } else if (nx > 1) {
        nx = 1 - (nx - 1)
        vxi = -Math.abs(vxi)
      }
      if (ny < -1) {
        ny = -1 + (-1 - ny)
        vyi = Math.abs(vyi)
      } else if (ny > 1) {
        ny = 1 - (ny - 1)
        vyi = -Math.abs(vyi)
      }
    }
    VX[i] = vxi
    VY[i] = vyi
    X[i] = nx
    Y[i] = ny
  }
  sim.lastMaxSpeed = Math.sqrt(maxSpeed2)
  sim.frame++
}

// ========================= Rendering =========================
function computeViewLayout(widthPx: number, heightPx: number): ViewLayout {
  const dpr = Math.max(1, (window.devicePixelRatio as number) || 1)
  const viewSize = Math.min(widthPx, heightPx)
  const viewX = Math.floor((widthPx - viewSize) / 2)
  const viewY = Math.floor((heightPx - viewSize) / 2)
  const scale = viewSize / WORLD_SIZE
  return {
    width: widthPx,
    height: heightPx,
    dpr,
    viewX,
    viewY,
    viewSize,
    scale,
  }
}

function setupOverlayCanvas(
  canvas: HTMLCanvasElement,
  layout: ViewLayout
): CanvasRenderingContext2D {
  canvas.style.width = `${layout.width}px`
  canvas.style.height = `${layout.height}px`
  canvas.width = Math.floor(layout.width * layout.dpr)
  canvas.height = Math.floor(layout.height * layout.dpr)
  const ctx = canvas.getContext("2d", { alpha: true })!
  ctx.setTransform(layout.dpr, 0, 0, layout.dpr, 0, 0)
  return ctx
}

function worldToScreen(x: number, y: number, layout: ViewLayout) {
  const sx = layout.viewX + (x + 1) * 0.5 * layout.viewSize
  const sy = layout.viewY + (y + 1) * 0.5 * layout.viewSize
  return [sx, sy] as const
}

function packInterleaved(sim: Sim) {
  const N = Math.min(sim.spec.N, sim.x.length, sim.interleaved.length / 3)
  const u = sim.interleaved
  for (let i = 0; i < N; i++) {
    const o = i * 3
    u[o] = sim.x[i]
    u[o + 1] = sim.y[i]
    u[o + 2] = sim.type[i]
  }
}

function drawOverlay(
  sim: Sim,
  ctx: CanvasRenderingContext2D,
  layout: ViewLayout,
  gpuPhysics: boolean
) {
  const { width, height, viewX, viewY, viewSize, scale } = layout
  const { showVel, showGrid } = sim.spec.overlays
  const N = Math.min(sim.spec.N, sim.x.length)

  ctx.clearRect(0, 0, width, height)

  if (showGrid) {
    ctx.save()
    ctx.beginPath()
    ctx.rect(viewX, viewY, viewSize, viewSize)
    ctx.clip()

    ctx.strokeStyle = "rgba(255,255,255,0.06)"
    ctx.lineWidth = 1
    const step = (sim.spec.cellSize * viewSize) / WORLD_SIZE

    for (let gx = viewX; gx <= viewX + viewSize + 0.5; gx += step) {
      ctx.beginPath()
      ctx.moveTo(gx, viewY)
      ctx.lineTo(gx, viewY + viewSize)
      ctx.stroke()
    }
    for (let gy = viewY; gy <= viewY + viewSize + 0.5; gy += step) {
      ctx.beginPath()
      ctx.moveTo(viewX, gy)
      ctx.lineTo(viewX + viewSize, gy)
      ctx.stroke()
    }
    ctx.restore()
  }

  if (showVel && N > 0) {
    const maxVelDraw = 80_000
    const step = Math.max(1, Math.ceil(N / maxVelDraw))
    ctx.strokeStyle = "rgba(255,255,255,0.35)"
    ctx.lineWidth = 1
    for (let i = 0; i < N; i += step) {
      const [px, py] = worldToScreen(sim.x[i], sim.y[i], layout)
      ctx.beginPath()
      ctx.moveTo(px, py)
      ctx.lineTo(px + sim.vx[i] * scale * 0.15, py + sim.vy[i] * scale * 0.15)
      ctx.stroke()
    }
  }

  ctx.fillStyle = "#FFFFFF"
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
  const A0 = sim.A[0]
    ?.slice(0, Math.min(6, sim.K))
    .map((v) => v.toFixed(2))
    .join(", ")
  const lines = [
    `N=${sim.spec.N}  K=${sim.K}  frame=${sim.frame}  WebGL${
      gpuPhysics ? " · WebGPU physics" : " · CPU physics"
    }`,
    `dt=${sim.spec.dt.toFixed(3)}  vMax=${sim.spec.vMax.toFixed(
      2
    )}  friction=${sim.spec.friction.toFixed(2)}  wrap=${sim.spec.wrap ? 1 : 0}`,
    `r̂Min=${sim.spec.rMin.toFixed(2)}  R̂=${sim.spec.R.toFixed(
      2
    )}  Rmax=${maxRMx(sim.RMx).toFixed(2)}  cell=${sim.spec.cellSize.toFixed(3)}`,
    `mutual=${sim.spec.mutualOnly ? 1 : 0}  settle=${
      sim.spec.settleEnabled ? 1 : 0
    }  k=${sim.spec.settleK.toFixed(2)}  sR=${sim.spec.settleR.toFixed(2)}`,
    `A[0,*]=[${A0 ?? ""}]`,
  ]
  let ty = viewY + 16
  for (const ln of lines) {
    ctx.fillText(ln, viewX + 8, ty)
    ty += 14
  }

  ctx.strokeStyle = "rgba(255,255,255,0.08)"
  ctx.strokeRect(viewX + 0.5, viewY + 0.5, viewSize - 1, viewSize - 1)
}

// ========================= Matrix Toolbar =========================
const RADIUS_STEP_MIN = 0.002
const RADIUS_STEP_MAX = 0.005
const RADIUS_PAIR_EPS = 0.002

/**
 * Editable radius bounds in the matrix panel (clamp + randomize).
 * Keep MATRIX_RMAX_HI modest vs WORLD_SIZE (2): large R balloons spatial-hash
 * neighbor reach (~ceil(maxR/cellSize) cells per axis) and dominates CPU cost.
 */
const MATRIX_RMIN_LO = 0.01
const MATRIX_RMIN_HI = 0.06
const MATRIX_RMAX_LO = 0.03
const MATRIX_RMAX_HI = 0.12

/** Random per-pair radii (Min r tab); scales like ring preset via spec scalars. */
function genRandomRMinTabMatrices(
  K: number,
  rng: () => number,
  scalarRMin: number,
  scalarR: number
): { rMinMx: number[][]; RMx: number[][] } {
  const rMinLo = clamp(
    scalarRMin * 0.4,
    MATRIX_RMIN_LO,
    MATRIX_RMIN_HI * 0.85
  )
  const rMinHi = clamp(
    scalarRMin * 2.2,
    rMinLo + 0.006,
    MATRIX_RMIN_HI
  )
  const rMaxCap = clamp(
    scalarR * 2.25,
    MATRIX_RMAX_LO + RADIUS_PAIR_EPS,
    MATRIX_RMAX_HI
  )

  const rMinMx: number[][] = Array.from({ length: K }, () => Array(K).fill(0))
  const RMx: number[][] = Array.from({ length: K }, () => Array(K).fill(0))
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) {
      const rmin = randRange(rng, rMinLo, rMinHi)
      rMinMx[i][j] = rmin
      const lo = Math.max(MATRIX_RMAX_LO, rmin + RADIUS_PAIR_EPS)
      const hi = clamp(rMaxCap, lo + 1e-4, MATRIX_RMAX_HI)
      RMx[i][j] = randRange(rng, lo, hi)
    }
  }
  return { rMinMx, RMx }
}

/** Random attraction cutoffs (Max r tab) for fixed min radii. */
function genRandomRMaxTabMatrix(
  K: number,
  rMinMx: number[][],
  rng: () => number,
  scalarR: number
): number[][] {
  const rMaxCap = clamp(
    scalarR * 2.25,
    MATRIX_RMAX_LO + RADIUS_PAIR_EPS,
    MATRIX_RMAX_HI
  )
  return Array.from({ length: K }, (_, i) =>
    Array.from({ length: K }, (_, j) => {
      const lo = Math.max(MATRIX_RMAX_LO, rMinMx[i][j] + RADIUS_PAIR_EPS)
      const hi = clamp(rMaxCap, lo + 1e-4, MATRIX_RMAX_HI)
      return randRange(rng, lo, hi)
    })
  )
}

/** Clamp LS / legacy matrices so spatial-hash reach stays reasonable (see MATRIX_RMAX_HI). */
function sanitizeRadiusMatrices(
  rMinMx: number[][],
  RMx: number[][]
): { rMinMx: number[][]; RMx: number[][] } {
  const K = rMinMx.length
  const r1 = cloneMx(rMinMx)
  const r2 = cloneMx(RMx)
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) {
      const rmin = clamp(r1[i][j], MATRIX_RMIN_LO, MATRIX_RMIN_HI)
      r1[i][j] = rmin
      let rmax = clamp(r2[i][j], MATRIX_RMAX_LO, MATRIX_RMAX_HI)
      rmax = Math.max(rmax, rmin + RADIUS_PAIR_EPS)
      rmax = Math.min(rmax, MATRIX_RMAX_HI)
      r2[i][j] = rmax
    }
  }
  return { rMinMx: r1, RMx: r2 }
}

function radiusToSwatch(lo: number, hi: number, v: number): string {
  const t = clamp((v - lo) / Math.max(1e-6, hi - lo), 0, 1)
  const g = Math.round(80 + 120 * t)
  const b = Math.round(100 + 100 * t)
  return `rgb(20, ${g}, ${b})`
}

/** Expand numeric range slightly so a nearly-flat matrix still shows contrast. */
function minMaxMxPad(
  m: number[][],
  fallbackLo: number,
  fallbackHi: number
): { lo: number; hi: number } {
  let lo = Infinity
  let hi = -Infinity
  for (const row of m) {
    for (const v of row) {
      if (Number.isFinite(v)) {
        if (v < lo) lo = v
        if (v > hi) hi = v
      }
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo + 1e-7) {
    const hi2 = Math.max(fallbackHi, fallbackLo + 1e-3)
    return { lo: fallbackLo, hi: hi2 }
  }
  const span = hi - lo
  const pad = span * 0.12 + 1e-4
  return { lo: lo - pad, hi: hi + pad }
}

type MatrixTab = "forces" | "rmin" | "rmax"

type MatrixToolbarProps = {
  K: number
  A: number[][]
  rMinMx: number[][]
  RMx: number[][]
  onChangeA: (A: number[][]) => void
  onApplyRadii: (rMinMx: number[][], RMx: number[][]) => void
  onRingPreset: () => void
  colors: string[]
  /** Fallback scalars for Randomize / Clear on radius tabs (current spec presets). */
  scalarRMin: number
  scalarR: number
}

function valueToSwatch(v: number): string {
  // map -1..1 to blue → black → red
  const t = (v + 1) / 2 // 0..1
  let r, g, b
  if (t < 0.5) {
    const f = t / 0.5
    r = 0
    g = 0
    b = Math.round(255 * (1 - f))
  } else {
    const f = (t - 0.5) / 0.5
    r = Math.round(255 * f)
    g = 0
    b = 0
  }
  return `rgb(${r}, ${g}, ${b})`
}

function cycleForceValue(v: number, dir: number): number {
  const steps = [-1, 0, 1]
  let best = steps[0]!
  let bestD = Math.abs(v - best)
  for (let s = 1; s < steps.length; s++) {
    const t = steps[s]!
    const d = Math.abs(v - t)
    if (d < bestD) {
      bestD = d
      best = t
    }
  }
  const idx = steps.indexOf(best)
  if (idx < 0) return 0
  return steps[(idx + dir + steps.length) % steps.length]!
}
const cellBtn: React.CSSProperties = {
  width: 22,
  height: 22,
  borderRadius: 4,
  border: "1px solid #222",
  cursor: "pointer",
  display: "block",
  padding: 0,
  margin: 0,
  appearance: "none",
  WebkitAppearance: "none",
  boxSizing: "border-box",
}

function MatrixToolbar({
  K,
  A,
  rMinMx,
  RMx,
  onChangeA,
  onApplyRadii,
  onRingPreset,
  colors,
  scalarRMin,
  scalarR,
}: MatrixToolbarProps) {
  const [tab, setTab] = useState<MatrixTab>("forces")

  const { lo: rMinSwLo, hi: rMinSwHi } = minMaxMxPad(
    rMinMx,
    MATRIX_RMIN_LO,
    MATRIX_RMIN_HI
  )
  const { lo: rMaxSwLo, hi: rMaxSwHi } = minMaxMxPad(
    RMx,
    MATRIX_RMAX_LO,
    MATRIX_RMAX_HI
  )

  const matrixOk =
    K >= 2 &&
    A.length === K &&
    !A.some((r) => r.length !== K) &&
    rMinMx.length === K &&
    !rMinMx.some((r) => r.length !== K) &&
    RMx.length === K &&
    !RMx.some((r) => r.length !== K)

  function handleClick(i: number, j: number, e: React.MouseEvent) {
    e.preventDefault()
    e.stopPropagation()
    const dir = e.type === "contextmenu" || e.button === 2 ? -1 : +1

    if (tab === "forces") {
      const nv = cycleForceValue(A[i][j], dir)
      const NA = A.map((row, ri) =>
        ri === i ? row.map((x, cj) => (cj === j ? clamp(nv, -1, 1) : x)) : row
      )
      onChangeA(NA)
      return
    }

    if (tab === "rmin") {
      const nr = clamp(
        rMinMx[i][j] + dir * RADIUS_STEP_MIN,
        MATRIX_RMIN_LO,
        MATRIX_RMIN_HI
      )
      const nextRMin = rMinMx.map((row, ri) =>
        ri === i ? row.map((x, cj) => (cj === j ? nr : x)) : row.slice()
      )
      const nextRMax = RMx.map((row, ri) =>
        ri === i
          ? row.map((R, cj) =>
              cj === j ? Math.max(R, nr + RADIUS_PAIR_EPS) : R
            )
          : row.slice()
      )
      onApplyRadii(nextRMin, nextRMax)
      return
    }

    const nR = clamp(
      RMx[i][j] + dir * RADIUS_STEP_MAX,
      MATRIX_RMAX_LO,
      MATRIX_RMAX_HI
    )
    const lo = rMinMx[i][j] + RADIUS_PAIR_EPS
    const nR2 = Math.max(nR, lo)
    const nextRMax = RMx.map((row, ri) =>
      ri === i ? row.map((R, cj) => (cj === j ? nR2 : R)) : row.slice()
    )
    onApplyRadii(rMinMx, nextRMax)
  }

  return (
    <div
      onContextMenu={(e) => {
        e.preventDefault()
        e.stopPropagation()
      }}
      style={{
        position: "fixed",
        right: 12,
        top: 64,
        zIndex: 100,
        isolation: "isolate",
        background: "#121212",
        border: "1px solid #222",
        borderRadius: 8,
        padding: 12,
        color: "#e5e7eb",
        maxHeight: "80vh",
        overflow: "auto",
        boxShadow: "0 8px 24px rgba(0,0,0,0.35)",
      }}
    >
      {!matrixOk ? (
        <div style={{ fontSize: 12, opacity: 0.85, maxWidth: 260 }}>
          Rules matrix is resizing ({K}×{K}) … reload preset or wait a moment.
        </div>
      ) : (
        <>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 8,
          flexWrap: "wrap",
        }}
      >
        <strong>Interaction matrix</strong>
        <button
          type="button"
          onClick={onRingPreset}
          style={{ ...chipStyle, padding: "4px 8px" }}
          title="Ring forces + ring-appropriate min/max radii"
        >
          Ring preset
        </button>
        <button
          type="button"
          onClick={() => {
            const rng = mulberry32(Date.now())
            if (tab === "forces") {
              onChangeA(genRandomMatrix(K, rng))
            } else if (tab === "rmin") {
              const { rMinMx: nr, RMx: nR } = genRandomRMinTabMatrices(
                K,
                rng,
                scalarRMin,
                scalarR
              )
              onApplyRadii(nr, nR)
            } else {
              onApplyRadii(
                cloneMx(rMinMx),
                genRandomRMaxTabMatrix(K, rMinMx, rng, scalarR)
              )
            }
          }}
          title="Randomize the current tab (forces or radii)"
          style={{ ...chipStyle, padding: "4px 8px" }}
        >
          Randomize
        </button>
        <button
          type="button"
          onClick={() => {
            if (tab === "forces") {
              onChangeA(Array.from({ length: K }, () => Array(K).fill(0)))
            } else if (tab === "rmin") {
              onApplyRadii(
                Array.from({ length: K }, () => Array(K).fill(scalarRMin)),
                Array.from({ length: K }, () => Array(K).fill(scalarR))
              )
            } else {
              onApplyRadii(
                cloneMx(rMinMx),
                Array.from({ length: K }, () => Array(K).fill(scalarR))
              )
            }
          }}
          style={{ ...chipStyle, padding: "4px 8px" }}
        >
          Clear
        </button>
      </div>

      <div
        style={{
          display: "flex",
          gap: 6,
          marginBottom: 10,
        }}
      >
        {(
          [
            ["forces", "Forces"],
            ["rmin", "Min r"],
            ["rmax", "Max r"],
          ] as const
        ).map(([id, label]) => (
          <button
            type="button"
            key={id}
            onClick={() => setTab(id)}
            style={{
              ...chipStyle,
              padding: "4px 10px",
              border:
                tab === id ? "1px solid #34d399" : "1px solid #374151",
              background: tab === id ? "#14532d" : "#111827",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      <div style={{ fontSize: 11, opacity: 0.8, marginBottom: 8 }}>
        {tab === "forces"
          ? "Left / right-click: cycle force weight −1, 0, +1."
          : tab === "rmin"
            ? "Left / right-click: step minimum interaction radius (hard shell)."
            : "Left / right-click: step maximum radius (attraction cutoff). Always ≥ min r + ε."}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: `24px repeat(${K}, 24px)`,
          gap: 4,
          alignItems: "center",
        }}
      >
        <div />
        {Array.from({ length: K }, (_, j) => (
          <div
            key={`h${j}`}
            title={`col ${j}`}
            style={{
              width: 22,
              height: 22,
              borderRadius: "50%",
              background: colors[j % colors.length],
              border: "1px solid #222",
            }}
          />
        ))}

        {Array.from({ length: K }, (_, i) => (
          <React.Fragment key={`row${i}`}>
            <div
              title={`row ${i}`}
              style={{
                width: 22,
                height: 22,
                borderRadius: "50%",
                background: colors[i % colors.length],
                border: "1px solid #222",
              }}
            />
            {Array.from({ length: K }, (_, j) => {
              let title = ""
              let bg = "transparent"
              if (tab === "forces") {
                title = `A[${i}][${j}] = ${A[i][j].toFixed(2)}`
                bg =
                  Math.abs(A[i][j]) > 1e-6
                    ? valueToSwatch(A[i][j])
                    : "transparent"
              } else if (tab === "rmin") {
                title = `rMin[${i}][${j}] = ${rMinMx[i][j].toFixed(3)}`
                bg = radiusToSwatch(rMinSwLo, rMinSwHi, rMinMx[i][j])
              } else {
                title = `R[${i}][${j}] = ${RMx[i][j].toFixed(3)}`
                bg = radiusToSwatch(rMaxSwLo, rMaxSwHi, RMx[i][j])
              }
              return (
                <div key={`c${i}-${j}`} style={{ position: "relative" }}>
                  <button
                    type="button"
                    onClick={(e) => handleClick(i, j, e)}
                    onContextMenu={(e) => handleClick(i, j, e)}
                    title={title}
                    aria-label={title}
                    style={{
                      ...cellBtn,
                      background: bg,
                      border: "1px solid #ffffff",
                    }}
                  />
                </div>
              )
            })}
          </React.Fragment>
        ))}
      </div>
        </>
      )}
    </div>
  )
}

// ========================= Hooks =========================
/** WebGL canvas + transparent 2D overlay; square viewport layout in shared `layoutRef`. */
function useRendererViewport(simRef: React.MutableRefObject<Sim | null>) {
  const glCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const layoutRef = useRef<ViewLayout | null>(null)
  const particleGLRef = useRef<ParticleGL | null>(null)
  const overlayCtxRef = useRef<CanvasRenderingContext2D | null>(null)

  useEffect(() => {
    const refresh = () => {
      const glCanvas = glCanvasRef.current
      const overlay = overlayCanvasRef.current
      const container = containerRef.current
      if (!glCanvas || !overlay || !container) return
      const rect = container.getBoundingClientRect()
      const layout = computeViewLayout(rect.width, rect.height)
      layoutRef.current = layout

      resizeDrawingSurface(glCanvas, layout)
      overlayCtxRef.current = setupOverlayCanvas(overlay, layout)

      try {
        if (!particleGLRef.current) {
          particleGLRef.current = createParticleGL(
            glCanvas,
            MAX_PARTICLES + 50_000
          )
        }
        const pr = particleGLRef.current
        const cap = Math.max(
          4096,
          Math.min(
            MAX_PARTICLES + 50_000,
            simRef.current?.spec.N ?? MAX_PARTICLES
          )
        )
        ensureParticleBufferCapacity(pr, cap)
      } catch (e) {
        console.error(e)
      }
    }

    const ro = new ResizeObserver(() => refresh())
    if (containerRef.current) ro.observe(containerRef.current)
    refresh()
    return () => ro.disconnect()
  }, [simRef])

  return {
    glCanvasRef,
    overlayCanvasRef,
    containerRef,
    layoutRef,
    particleGLRef,
    overlayCtxRef,
  }
}

// ========================= Component =========================
export default function App() {
  // Use SPEC directly; do not override A on first render.
  const [spec, setSpec] = useState<Spec>({ ...SPEC })
  const [seed, setSeed] = useState<number>(SPEC.seed)
  const [paused, setPaused] = useState(false)
  const [showRules, setShowRules] = useState(true)
  const [fps, setFps] = useState(0)

  const simRef = useRef<Sim | null>(null)
  const gpuRunnerRef = useRef<GpuSimRunner | null>(null)
  const [gpuPhysics, setGpuPhysics] = useState(false)

  const {
    glCanvasRef,
    overlayCanvasRef,
    containerRef,
    layoutRef,
    particleGLRef,
    overlayCtxRef,
  } = useRendererViewport(simRef)

  // -------- Resolve matrices when K changes or genMatrix is set.
  useEffect(() => {
    const need =
      spec.genMatrix ||
      spec.A.length !== spec.K ||
      spec.A.some((r) => r.length !== spec.K) ||
      spec.rMinMx.length !== spec.K ||
      spec.rMinMx.some((r) => r.length !== spec.K) ||
      spec.RMx.length !== spec.K ||
      spec.RMx.some((r) => r.length !== spec.K)

    if (!need) return

    const loaded = loadRulesFromLS(spec.K, spec.rMin, spec.R)
    if (loaded) {
      const sane = sanitizeRadiusMatrices(loaded.rMinMx, loaded.RMx)
      setSpec((s) => ({
        ...s,
        A: cloneMx(loaded.A),
        rMinMx: sane.rMinMx,
        RMx: sane.RMx,
        genMatrix: false,
      }))
      return
    }

    const A = genRingPreset(spec.K)
    const { rMinMx, RMx } = genRingRadiusPreset(spec.K, spec.rMin, spec.R)
    setSpec((s) => ({
      ...s,
      A,
      rMinMx,
      RMx,
      genMatrix: false,
    }))
  }, [spec.K, spec.genMatrix])

  // -------- WebGPU physics (optional; depends on cellSize for grid buffer cap).
  useEffect(() => {
    let cancelled = false
    setGpuPhysics(false)
    if (!ENABLE_WEBGPU_PHYSICS) return
    const gridDim = Math.max(1, Math.ceil(WORLD_SIZE / spec.cellSize))
    void (async () => {
      const runner = await createGpuSimRunner(MAX_PARTICLES + 50_000, gridDim)
      if (cancelled) {
        runner?.dispose()
        return
      }
      gpuRunnerRef.current?.dispose()
      gpuRunnerRef.current = runner
      const sim = simRef.current
      if (runner && sim) {
        runner.uploadParticleState(sim)
        setGpuPhysics(true)
      } else {
        setGpuPhysics(false)
      }
    })()
    return () => {
      cancelled = true
      gpuRunnerRef.current?.dispose()
      gpuRunnerRef.current = null
    }
  }, [spec.cellSize])

  // -------- Initialize / Re-initialize simulation when core layout changes.
  useEffect(() => {
    simRef.current = initSim(spec, seed)
    gpuRunnerRef.current?.uploadParticleState(simRef.current)
  }, [
    seed,
    spec.N,
    spec.K,
    spec.cellSize,
    spec.wrap,
    spec.rMin,
    spec.R,
    spec.dt,
    spec.friction,
    spec.vMax,
  ])

  // keep runtime sim reading latest spec without forcing reset
  useEffect(() => {
    if (simRef.current) simRef.current.spec = spec
  }, [spec])

  // draw + sim loop. Sync; WebGPU readback is fire-and-forget.
  useEffect(() => {
    let stopped = false
    let raf = 0

    const tick = () => {
      if (stopped) return
      const sim = simRef.current
      const layout = layoutRef.current
      const pr = particleGLRef.current
      const octx = overlayCtxRef.current
      if (!sim || !layout || !pr || !octx) {
        raf = requestAnimationFrame(tick)
        return
      }

      if (!paused) {
        const gpu = gpuRunnerRef.current
        if (gpu && gpuPhysics) {
          // Grid is built on GPU; skip CPU rebuildGrid and avoid uploads.
          gpu.step(sim)
        } else {
          step(sim)
        }
      }

      const n = Math.min(sim.spec.N, sim.x.length)
      ensureParticleBufferCapacity(pr, n)
      packInterleaved(sim)
      drawParticles(
        pr,
        sim.interleaved,
        n,
        layout,
        sim.K,
        TYPE_COLORS,
        n > 400_000 ? 1.2 : 1.75
      )
      drawOverlay(sim, octx, layout, gpuPhysics)

      raf = requestAnimationFrame(tick)
    }

    raf = requestAnimationFrame(tick)
    return () => {
      stopped = true
      cancelAnimationFrame(raf)
    }
  }, [paused, gpuPhysics])

  // FPS meter
  useEffect(() => {
    let last = performance.now(),
      frames = 0,
      raf = 0 as unknown as number
    function tick() {
      frames++
      const now = performance.now()
      if (now - last >= 500) {
        setFps((frames * 1000) / (now - last))
        frames = 0
        last = now
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  // persist interaction rules
  useEffect(() => {
    if (
      spec.A.length === spec.K &&
      !spec.A.some((r) => r.length !== spec.K) &&
      spec.rMinMx.length === spec.K &&
      !spec.rMinMx.some((r) => r.length !== spec.K) &&
      spec.RMx.length === spec.K &&
      !spec.RMx.some((r) => r.length !== spec.K)
    ) {
      saveRulesToLS(spec.K, spec.A, spec.rMinMx, spec.RMx)
    }
  }, [spec.A, spec.rMinMx, spec.RMx, spec.K])

  function applyMatrix(A: number[][]) {
    setSpec((s) => ({ ...s, A, genMatrix: false }))
    if (simRef.current && simRef.current.K === A.length) {
      simRef.current.A = cloneMx(A)
    }
    gpuRunnerRef.current?.markRulesDirty()
  }

  function applyRadii(rMinMx: number[][], RMx: number[][]) {
    setSpec((s) => ({
      ...s,
      rMinMx: cloneMx(rMinMx),
      RMx: cloneMx(RMx),
      genMatrix: false,
    }))
    const sim = simRef.current
    if (sim && sim.K === rMinMx.length && sim.K === RMx.length) {
      sim.rMinMx = cloneMx(rMinMx)
      sim.RMx = cloneMx(RMx)
    }
    gpuRunnerRef.current?.markRulesDirty()
  }

  // handlers
  const togglePause = () => setPaused((p) => !p)
  const handleReset = () => {
    simRef.current = initSim(spec, seed)
    gpuRunnerRef.current?.uploadParticleState(simRef.current)
  }
  const handleRandomizeSeed = () => setSeed(Math.floor(Math.random() * 1e9))
  const incN = (delta: number) => {
    const newN = clamp(spec.N + delta, 0, MAX_PARTICLES)
    setSpec((s) => ({ ...s, N: newN }))
  }
  const incK = (delta: number) => {
    const maxK = TYPE_COLORS.length
    const newK = clamp(spec.K + delta, 2, maxK)
    setSpec((s) => ({ ...s, K: newK, genMatrix: true }))
  }

  const applyRingPreset = () => {
    const A = genRingPreset(spec.K)
    const { rMinMx, RMx } = genRingRadiusPreset(spec.K, spec.rMin, spec.R)
    setSpec((s) => ({
      ...s,
      A,
      rMinMx,
      RMx,
      genMatrix: false,
    }))
    const sim = simRef.current
    if (sim && sim.K === spec.K) {
      sim.A = cloneMx(A)
      sim.rMinMx = cloneMx(rMinMx)
      sim.RMx = cloneMx(RMx)
    }
    gpuRunnerRef.current?.markRulesDirty()
  }

  const inc = (v: number, d: number, lo: number, hi: number) =>
    clamp(v + d, lo, hi)

  return (
    <div
      style={{
        display: "grid",
        gridTemplateRows: "auto 1fr auto",
        minHeight: "100vh",
        background: "#0a0a0a",
        color: "#eaeaea",
        fontFamily:
          "ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial",
      }}
    >
      {/* Header */}
      <div style={{ padding: "12px 16px", borderBottom: "1px solid #222" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <strong>Particle Interaction Prototype · PL + Rules</strong>
          <span style={{ opacity: 0.7 }}>FPS: {fps.toFixed(0)}</span>
          <button
            onClick={togglePause}
            style={buttonStyle}
            title="Pause or resume"
          >
            {paused ? "Resume" : "Pause"}
          </button>
          <button
            onClick={handleReset}
            style={buttonStyle}
            title="Reset with current seed/spec"
          >
            Reset
          </button>
          <button
            onClick={handleRandomizeSeed}
            style={buttonStyle}
            title="Randomize seed and reset"
          >
            New Seed
          </button>

          <div style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
            <span style={{ opacity: 0.7 }}>N:</span>
            <button onClick={() => incN(-100)} style={chipStyle}>
              −100
            </button>
            <span>{spec.N}</span>
            <button onClick={() => incN(+100)} style={chipStyle}>
              +100
            </button>
          </div>

          <div style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
            <span style={{ opacity: 0.7 }}>K:</span>
            <button onClick={() => incK(-1)} style={chipStyle}>
              −1
            </button>
            <span>{spec.K}</span>
            <button onClick={() => incK(+1)} style={chipStyle}>
              +1
            </button>
          </div>

          {/* Overlay toggles */}
          <div
            style={{
              display: "inline-flex",
              gap: 8,
              alignItems: "center",
              marginLeft: 12,
            }}
          >
            <label
              style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
              title="Draw velocity vectors"
            >
              <input
                type="checkbox"
                checked={spec.overlays.showVel}
                onChange={(e) =>
                  setSpec({
                    ...spec,
                    overlays: { ...spec.overlays, showVel: e.target.checked },
                  })
                }
              />
              Vel
            </label>
            <label
              style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
              title="Show spatial grid"
            >
              <input
                type="checkbox"
                checked={spec.overlays.showGrid}
                onChange={(e) =>
                  setSpec({
                    ...spec,
                    overlays: { ...spec.overlays, showGrid: e.target.checked },
                  })
                }
              />
              Grid
            </label>
            <label
              style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
              title="World wraps on edges"
            >
              <input
                type="checkbox"
                checked={spec.wrap}
                onChange={(e) => setSpec({ ...spec, wrap: e.target.checked })}
              />
              Wrap
            </label>
          </div>

          {/* Behavior toggles */}
          <div
            style={{
              display: "inline-flex",
              gap: 10,
              alignItems: "center",
              marginLeft: 12,
            }}
          >
            <label
              style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
              title="Only allow attraction when A[i][j] and A[j][i] are both > 0"
            >
              <input
                type="checkbox"
                checked={spec.mutualOnly}
                onChange={(e) =>
                  setSpec({ ...spec, mutualOnly: e.target.checked })
                }
              />
              Mutual
            </label>
            <label
              style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
              title="Enable settling (radial damping for mutually attracted pairs)"
            >
              <input
                type="checkbox"
                checked={spec.settleEnabled}
                onChange={(e) =>
                  setSpec({ ...spec, settleEnabled: e.target.checked })
                }
              />
              Settle
            </label>

            <div
              style={{ display: "inline-flex", gap: 6, alignItems: "center" }}
            >
              <span style={{ opacity: 0.7 }}>k:</span>
              <button
                onClick={() =>
                  setSpec((s) => ({
                    ...s,
                    settleK: inc(s.settleK, -0.01, 0, 20),
                  }))
                }
                style={chipStyle}
              >
                −
              </button>
              <span>{spec.settleK.toFixed(2)}</span>
              <button
                onClick={() =>
                  setSpec((s) => ({
                    ...s,
                    settleK: inc(s.settleK, +0.01, 0, 20),
                  }))
                }
                style={chipStyle}
              >
                +
              </button>
            </div>

            <div
              style={{ display: "inline-flex", gap: 6, alignItems: "center" }}
            >
              <span style={{ opacity: 0.7 }} title="Velocity damping per step">
                friction:
              </span>
              <button
                onClick={() =>
                  setSpec((s) => ({
                    ...s,
                    friction: inc(s.friction, -0.02, 0, 1.5),
                  }))
                }
                style={chipStyle}
              >
                −
              </button>
              <span>{spec.friction.toFixed(2)}</span>
              <button
                onClick={() =>
                  setSpec((s) => ({
                    ...s,
                    friction: inc(s.friction, +0.02, 0, 1.5),
                  }))
                }
                style={chipStyle}
              >
                +
              </button>
            </div>

            <div
              style={{ display: "inline-flex", gap: 6, alignItems: "center" }}
            >
              <span style={{ opacity: 0.7 }}>sR:</span>
              <button
                onClick={() =>
                  setSpec((s) => ({
                    ...s,
                    settleR: inc(s.settleR, -0.01, 0.05, 1),
                  }))
                }
                style={chipStyle}
              >
                −
              </button>
              <span>{spec.settleR.toFixed(2)}</span>
              <button
                onClick={() =>
                  setSpec((s) => ({
                    ...s,
                    settleR: inc(s.settleR, +0.01, 0.05, 1),
                  }))
                }
                style={chipStyle}
              >
                +
              </button>
            </div>
          </div>

          <button
            onClick={() => setShowRules((s) => !s)}
            style={chipStyle}
            title="Show/Hide rules editor"
          >
            {showRules ? "Hide Rules" : "Show Rules"}
          </button>
        </div>
      </div>

      {/* Canvas area (middle row) */}
      <div
        ref={containerRef}
        style={{
          position: "relative",
          width: "100%",
          height: "100%",
          minHeight: 0,
          overflow: "hidden",
        }}
      >
        <canvas
          ref={glCanvasRef}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            display: "block",
          }}
        />
        <canvas
          ref={overlayCanvasRef}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            display: "block",
            pointerEvents: "none",
          }}
        />
      </div>

      {/* Footer */}
      <div
        style={{
          padding: "10px 16px",
          borderTop: "1px solid #222",
          fontSize: 12,
          color: "#cfcfcf",
        }}
      >
        <span style={{ opacity: 0.85 }}>
          rMinPreset={spec.rMin.toFixed(2)} · Rpreset=
          {spec.R.toFixed(2)} · Rmax={maxRMx(spec.RMx).toFixed(2)} · dt=
          {spec.dt} · friction={spec.friction.toFixed(2)} · vMax={spec.vMax} ·
          cell={spec.cellSize} · seed={seed}
        </span>
      </div>

      {/* Rules toolbar */}
      {showRules && (
        <MatrixToolbar
          K={spec.K}
          A={spec.A}
          rMinMx={spec.rMinMx}
          RMx={spec.RMx}
          onChangeA={applyMatrix}
          onApplyRadii={applyRadii}
          onRingPreset={applyRingPreset}
          colors={TYPE_COLORS}
          scalarRMin={spec.rMin}
          scalarR={spec.R}
        />
      )}
    </div>
  )
}

// ========================= Styles =========================
const buttonStyle: React.CSSProperties = {
  background: "#1f2937",
  color: "#e5e7eb",
  border: "1px solid #374151",
  padding: "6px 10px",
  borderRadius: 6,
  cursor: "pointer",
}

const chipStyle: React.CSSProperties = {
  background: "#111827",
  color: "#e5e7eb",
  border: "1px solid #374151",
  padding: "2px 6px",
  borderRadius: 6,
  cursor: "pointer",
}
