import React, { useEffect, useMemo, useRef, useState } from "react"
import type { Sim, Spec } from "./simTypes"
import {
  createParticleGL,
  drawParticles,
  ensureParticleBufferCapacity,
  resizeDrawingSurface,
  defaultViewCamera,
  type ParticleGL,
  type ViewCamera,
  type ViewLayout,
} from "./webglParticles"
import { createGpuSimRunner, type GpuSimRunner } from "./webgpuSim"
import {
  OPEN_TRAIL_CHUNK_WORLD,
  openTrailVisibleChunkRange,
} from "./openTrailChunks"
import { OrganismTracker, type TrackerSnapshot } from "./tracker"
import { Competitors } from "./Competitors"

/**
 * Particle Life backend + editable ruleset (A matrix) with toolbar.
 * - World: vertical span [-1,1]; horizontal span scales with viewport aspect; optional wrap.
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
  "#CD853F", // peru
  "#DA70D6", // orchid
  "#20B2AA", // light sea green
  "#9370DB", // medium purple
  "#3CB371", // medium sea green
  "#F4A460", // sandy brown
  "#4169E1", // royal blue
  "#DC143C", // crimson
  "#2E8B57", // sea green
  "#FF1493", // deep pink
  "#00FA9A", // medium spring green
  "#8B4513", // saddle brown
  "#483D8B", // dark slate blue
  "#FFD700", // gold
  "#FF6347", // tomato
  "#40E0D0", // turquoise
  "#EE82EE", // violet (named)
  "#9ACD32", // yellow green
  "#D2691E", // chocolate
  "#5F9EA0", // cadet blue
  "#DDA0DD", // plum
  "#00FF7F", // spring green
  "#B22222", // fire brick
  "#48D1CC", // medium turquoise
  "#C71585", // medium violet red
  "#32CD32", // lime green
  "#FF4500", // orange red
  "#1E90FF", // dodger blue
  "#ADFF2F", // green yellow
  "#FF69B4", // hot pink
  "#BA55D3", // medium orchid
  "#F08080", // light coral
  "#7CFC00", // lawn green
]

/** Max particles (WebGL draws in one call; physics is still CPU spatial-hash). */
const MAX_PARTICLES = 1_000_000
const ENABLE_WEBGPU_PHYSICS = true
/** Cap open-world spatial hash dimensions so memory stays bounded if the cloud spreads. */
const OPEN_GRID_MAX_SIDE = 512

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
/** Vertical torus half-extent; horizontal follows view aspect ratio. */
const WORLD_HALF_H = 1
/** Initial spawn: uniform disk at world origin (tight blob vs full square). */
const INIT_CLUSTER_RADIUS = 0.09
/** Reference vertical span for force scaling vs pixel space. */
const REF_VERTICAL_SPAN = 2 * WORLD_HALF_H

/** Max width/height aspect for GPU grid buffer sizing (ultrawide displays). */
const MAX_VIEW_ASPECT = 48

function maxGridCellsForCellSize(cellSize: number): number {
  const maxWorldHalfW = MAX_VIEW_ASPECT * WORLD_HALF_H
  const worldWidth = 2 * maxWorldHalfW
  const worldHeight = REF_VERTICAL_SPAN
  const gx = Math.max(1, Math.ceil(worldWidth / cellSize))
  const gy = Math.max(1, Math.ceil(worldHeight / cellSize))
  return gx * gy
}

function foldPeriodic(pos: number, halfSpan: number, fullSpan: number): number {
  let p = pos
  while (p < -halfSpan) p += fullSpan
  while (p > halfSpan) p -= fullSpan
  return p
}

function syncSimWorldToLayout(sim: Sim, layout: ViewLayout) {
  if (layout.width < 2 || layout.height < 2) return
  const worldHalfW = (layout.width / layout.height) * WORLD_HALF_H
  const worldHalfH = WORLD_HALF_H
  if (sim.spec.wrap) {
    const worldWidth = 2 * worldHalfW
    const worldHeight = 2 * worldHalfH
    const gx = Math.max(1, Math.ceil(worldWidth / sim.spec.cellSize))
    const gy = Math.max(1, Math.ceil(worldHeight / sim.spec.cellSize))
    if (gx !== sim.gridDimX || gy !== sim.gridDimY) {
      sim.gridDimX = gx
      sim.gridDimY = gy
      sim.cellHead = new Int32Array(gx * gy)
    }
    const extentChanged =
      Math.abs(worldHalfW - sim.worldHalfW) > 1e-7 ||
      Math.abs(worldHalfH - sim.worldHalfH) > 1e-7
    sim.worldHalfW = worldHalfW
    sim.worldHalfH = worldHalfH
    if (extentChanged) {
      const N = Math.min(sim.spec.N, sim.x.length)
      for (let i = 0; i < N; i++) {
        sim.x[i] = foldPeriodic(sim.x[i], worldHalfW, worldWidth)
        sim.y[i] = foldPeriodic(sim.y[i], worldHalfH, worldHeight)
      }
    }
  } else {
    sim.worldHalfW = worldHalfW
    sim.worldHalfH = worldHalfH
  }
}

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

const DEFAULT_K = 30

const REF_WORLD_UNITS_PER_PX = REF_VERTICAL_SPAN / 1280

const LS_KEY_V2 = "pl_rules_v6"
const LS_KEY_V1 = "pl_rules_v1"

function saveRulesToLS(
  K: number,
  A: number[][],
  rMinMx: number[][],
  RMx: number[][]
) {
  try {
    localStorage.setItem(LS_KEY_V2, JSON.stringify({ K, A, rMinMx, RMx }))
  } catch {
    /* storage may be unavailable */
  }
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

type InitSimResult = {
  sim: Sim
  /** Merge into React `spec` when rules were randomly drawn (non-localStorage). */
  specPatch?: Pick<Spec, "A" | "rMinMx" | "RMx" | "genMatrix">
}

function initSim(
  spec: Spec,
  seedOverride?: number,
  opts?: { rerollInteractionRules?: boolean }
): InitSimResult {
  const reroll = opts?.rerollInteractionRules ?? false
  const rng = mulberry32(seedOverride ?? spec.seed)
  const K = spec.K

  // choose rules: localStorage → random (genMatrix / reroll) → spec matrices
  const loaded = loadRulesFromLS(K, spec.rMin, spec.R)
  let A: number[][]
  let rMinMx: number[][]
  let RMx: number[][]
  let usedRandomRules = false

  if (loaded) {
    A = loaded.A
    const sane = sanitizeRadiusMatrices(loaded.rMinMx, loaded.RMx)
    rMinMx = sane.rMinMx
    RMx = sane.RMx
  } else if (spec.genMatrix || reroll) {
    const rulesSalt =
      (Date.now() ^ ((seedOverride ?? spec.seed) * 2654435761)) >>> 0
    const rngRules = mulberry32(rulesSalt)
    A = genRandomMatrix(K, rngRules)
    const rm = genRandomRMinTabMatrices(K, rngRules, spec.rMin, spec.R)
    rMinMx = rm.rMinMx
    RMx = rm.RMx
    usedRandomRules = true
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
    const ang = rng() * 2 * Math.PI
    const rad = INIT_CLUSTER_RADIUS * Math.sqrt(rng())
    x[i] = rad * Math.cos(ang)
    y[i] = rad * Math.sin(ang)
    vx[i] = randRange(rng, -0.005, 0.005)
    vy[i] = randRange(rng, -0.005, 0.005)
    type[i] = Math.floor(rng() * K)
  }

  const worldHalfW = 1
  const worldHalfH = WORLD_HALF_H
  const next = new Int32Array(spec.N)

  let gridDimX: number
  let gridDimY: number
  let cellHead: Int32Array
  let gridOriginX: number
  let gridOriginY: number
  let gridCellEff: number

  if (spec.wrap) {
    const worldWidth = 2 * worldHalfW
    const worldHeight = 2 * worldHalfH
    gridDimX = Math.max(1, Math.ceil(worldWidth / spec.cellSize))
    gridDimY = Math.max(1, Math.ceil(worldHeight / spec.cellSize))
    cellHead = new Int32Array(gridDimX * gridDimY)
    gridOriginX = 0
    gridOriginY = 0
    gridCellEff = spec.cellSize
  } else {
    const mr = maxRMx(RMx)
    const cs0 = spec.cellSize
    const reach0 = Math.max(1, Math.ceil(mr / cs0))
    const pad0 = mr + reach0 * cs0 + cs0
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity
    for (let i = 0; i < spec.N; i++) {
      const xi = x[i],
        yi = y[i]
      if (xi < minX) minX = xi
      if (xi > maxX) maxX = xi
      if (yi < minY) minY = yi
      if (yi > maxY) maxY = yi
    }
    let spanX = maxX - minX + 2 * pad0
    let spanY = maxY - minY + 2 * pad0
    if (spanX < cs0) spanX = cs0
    if (spanY < cs0) spanY = cs0
    let cs = cs0
    let ncx = Math.max(1, Math.ceil(spanX / cs))
    let ncy = Math.max(1, Math.ceil(spanY / cs))
    if (ncx > OPEN_GRID_MAX_SIDE || ncy > OPEN_GRID_MAX_SIDE) {
      const scale = Math.max(
        ncx / OPEN_GRID_MAX_SIDE,
        ncy / OPEN_GRID_MAX_SIDE
      )
      cs = cs0 * scale
      ncx = Math.min(
        OPEN_GRID_MAX_SIDE,
        Math.max(1, Math.ceil(spanX / cs))
      )
      ncy = Math.min(
        OPEN_GRID_MAX_SIDE,
        Math.max(1, Math.ceil(spanY / cs))
      )
    }
    const midX = (minX + maxX) * 0.5
    const midY = (minY + maxY) * 0.5
    const coverX = ncx * cs
    const coverY = ncy * cs
    gridOriginX = midX - coverX * 0.5
    gridOriginY = midY - coverY * 0.5
    gridCellEff = cs
    gridDimX = ncx
    gridDimY = ncy
    cellHead = new Int32Array(ncx * ncy)
  }

  const specPatch: InitSimResult["specPatch"] = usedRandomRules
    ? {
        A: cloneMx(A),
        rMinMx: cloneMx(rMinMx),
        RMx: cloneMx(RMx),
        genMatrix: false,
      }
    : undefined

  return {
    sim: {
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
      worldHalfW,
      worldHalfH,
      gridDimX,
      gridDimY,
      cellHead,
      next,
      rng,
      frame: 0,
      lastMaxSpeed: 0,
      A: cloneMx(A),
      rMinMx: cloneMx(rMinMx),
      RMx: cloneMx(RMx),
      gridOriginX,
      gridOriginY,
      gridCellEff,
    },
    specPatch,
  }
}

// ========================= Neighbors =========================
function cellIndexOfTorus(
  x: number,
  y: number,
  worldHalfW: number,
  worldHalfH: number,
  worldWidth: number,
  worldHeight: number,
  gridDimX: number,
  gridDimY: number
): number {
  const gxx = clamp(
    Math.floor(((x + worldHalfW) / worldWidth) * gridDimX),
    0,
    gridDimX - 1
  )
  const gyy = clamp(
    Math.floor(((y + worldHalfH) / worldHeight) * gridDimY),
    0,
    gridDimY - 1
  )
  return gxx + gridDimX * gyy
}

function cellIndexOfOpen(
  x: number,
  y: number,
  ox: number,
  oy: number,
  cs: number,
  gridDimX: number,
  gridDimY: number
): number {
  const gxx = clamp(Math.floor((x - ox) / cs), 0, gridDimX - 1)
  const gyy = clamp(Math.floor((y - oy) / cs), 0, gridDimY - 1)
  return gxx + gridDimX * gyy
}

/** Resize/reanchor open-world grid to cover all particles (plus neighbor padding). */
function prepareOpenGrid(sim: Sim, N: number, maxR: number) {
  const cs0 = sim.spec.cellSize
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity
  for (let i = 0; i < N; i++) {
    const xi = sim.x[i],
      yi = sim.y[i]
    if (xi < minX) minX = xi
    if (xi > maxX) maxX = xi
    if (yi < minY) minY = yi
    if (yi > maxY) maxY = yi
  }
  if (!Number.isFinite(minX)) return

  const reachNorm = Math.max(1, Math.ceil(maxR / cs0))
  const pad = maxR + reachNorm * cs0 + cs0
  let spanX = maxX - minX + 2 * pad
  let spanY = maxY - minY + 2 * pad
  if (spanX < cs0) spanX = cs0
  if (spanY < cs0) spanY = cs0

  let cs = cs0
  let ncx = Math.max(1, Math.ceil(spanX / cs))
  let ncy = Math.max(1, Math.ceil(spanY / cs))
  if (ncx > OPEN_GRID_MAX_SIDE || ncy > OPEN_GRID_MAX_SIDE) {
    const scale = Math.max(
      ncx / OPEN_GRID_MAX_SIDE,
      ncy / OPEN_GRID_MAX_SIDE
    )
    cs = cs0 * scale
    ncx = Math.min(
      OPEN_GRID_MAX_SIDE,
      Math.max(1, Math.ceil(spanX / cs))
    )
    ncy = Math.min(
      OPEN_GRID_MAX_SIDE,
      Math.max(1, Math.ceil(spanY / cs))
    )
  }

  const midX = (minX + maxX) * 0.5
  const midY = (minY + maxY) * 0.5
  const coverX = ncx * cs
  const coverY = ncy * cs
  sim.gridOriginX = midX - coverX * 0.5
  sim.gridOriginY = midY - coverY * 0.5
  sim.gridCellEff = cs

  if (ncx !== sim.gridDimX || ncy !== sim.gridDimY) {
    sim.gridDimX = ncx
    sim.gridDimY = ncy
    sim.cellHead = new Int32Array(ncx * ncy)
  }
}

function rebuildGrid(sim: Sim) {
  sim.cellHead.fill(-1)
  // Safe N prevents writing past typed-array bounds after spec changes.
  const N = Math.min(sim.spec.N, sim.x.length, sim.next.length)
  if (sim.spec.wrap) {
    const { gridDimX, gridDimY, worldHalfW, worldHalfH } = sim
    const worldWidth = 2 * worldHalfW
    const worldHeight = 2 * worldHalfH
    for (let i = 0; i < N; i++) {
      const idx = cellIndexOfTorus(
        sim.x[i],
        sim.y[i],
        worldHalfW,
        worldHalfH,
        worldWidth,
        worldHeight,
        gridDimX,
        gridDimY
      )
      sim.next[i] = sim.cellHead[idx]
      sim.cellHead[idx] = i
    }
  } else {
    const { gridDimX, gridDimY, gridOriginX, gridOriginY, gridCellEff } = sim
    for (let i = 0; i < N; i++) {
      const idx = cellIndexOfOpen(
        sim.x[i],
        sim.y[i],
        gridOriginX,
        gridOriginY,
        gridCellEff,
        gridDimX,
        gridDimY
      )
      sim.next[i] = sim.cellHead[idx]
      sim.cellHead[idx] = i
    }
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

  const RMAXMX = sim.RMx
  const maxR = maxRMx(RMAXMX)
  if (!sp.wrap) {
    prepareOpenGrid(sim, N, maxR)
  }
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
  const cellHead = sim.cellHead
  const linkNext = sim.next
  const gdx = sim.gridDimX
  const gdy = sim.gridDimY
  const whw = sim.worldHalfW
  const whh = sim.worldHalfH
  const worldWidth = 2 * whw
  const worldHeight = 2 * whh
  const wrap = sp.wrap
  const gOx = sim.gridOriginX
  const gOy = sim.gridOriginY
  const gCs = sim.gridCellEff
  const cellSize = sp.cellSize
  const mutualOnly = sp.mutualOnly
  const repel = sp.repel
  const forceFactor = sp.forceFactor
  const settleEnabled = sp.settleEnabled
  const settleR = sp.settleR
  const settleK = sp.settleK
  const cCrit = 2 / dt
  const cSettle = Math.min(Math.max(settleK, 0), cCrit)

  const maxR2 = maxR * maxR
  const reach = Math.max(1, Math.ceil(maxR / (wrap ? cellSize : gCs)))

  FX.fill(0, 0, N)
  FY.fill(0, 0, N)

  for (let i = 0; i < N; i++) {
    const ti = TYPE[i] | 0
    const xi = X[i]
    const yi = Y[i]
    const ARi = A[ti]
    const RminRi = RMINMX[ti]
    const RmaxRi = RMAXMX[ti]

    let cx: number
    let cy: number
    if (wrap) {
      const cxi = ((xi + whw) / worldWidth) * gdx
      const cyi = ((yi + whh) / worldHeight) * gdy
      cx = cxi < 0 ? 0 : cxi >= gdx ? gdx - 1 : cxi | 0
      cy = cyi < 0 ? 0 : cyi >= gdy ? gdy - 1 : cyi | 0
    } else {
      const cxi = (xi - gOx) / gCs
      const cyi = (yi - gOy) / gCs
      cx = cxi < 0 ? 0 : cxi >= gdx ? gdx - 1 : cxi | 0
      cy = cyi < 0 ? 0 : cyi >= gdy ? gdy - 1 : cyi | 0
    }

    let lfx = 0
    let lfy = 0

    for (let dy = -reach; dy <= reach; dy++) {
      let ny = cy + dy
      if (wrap) {
        ny = ((ny % gdy) + gdy) % gdy
      } else if (ny < 0 || ny >= gdy) {
        continue
      }
      const rowBase = ny * gdx
      for (let dx = -reach; dx <= reach; dx++) {
        let nx = cx + dx
        if (wrap) {
          nx = ((nx % gdx) + gdx) % gdx
        } else if (nx < 0 || nx >= gdx) {
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
              if (ddx > worldWidth * 0.5) ddx -= worldWidth
              else if (ddx < -worldWidth * 0.5) ddx += worldWidth
              if (ddy > worldHeight * 0.5) ddy -= worldHeight
              else if (ddy < -worldHeight * 0.5) ddy += worldHeight
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

    const nx = X[i] + dt * vxi
    const ny = Y[i] + dt * vyi
    if (wrap) {
      let px = nx
      let py = ny
      while (px < -whw) px += worldWidth
      while (px > whw) px -= worldWidth
      while (py < -whh) py += worldHeight
      while (py > whh) py -= worldHeight
      X[i] = px
      Y[i] = py
    } else {
      X[i] = nx
      Y[i] = ny
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
  const dprRaw = (window.devicePixelRatio as number) || 1
  const dpr = Math.min(2, Math.max(1, dprRaw))
  const worldHalfH = WORLD_HALF_H
  const worldHalfW =
    heightPx > 0 ? (widthPx / heightPx) * worldHalfH : worldHalfH
  const worldWidth = 2 * worldHalfW
  const worldHeight = 2 * worldHalfH
  return {
    width: widthPx,
    height: heightPx,
    dpr,
    viewX: 0,
    viewY: 0,
    viewW: widthPx,
    viewH: heightPx,
    worldHalfW,
    worldHalfH,
    scaleX: widthPx / worldWidth,
    scaleY: heightPx / worldHeight,
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

const ZOOM_MIN = 0.12
const ZOOM_MAX = 80

function worldToScreen(
  x: number,
  y: number,
  layout: ViewLayout,
  cam: ViewCamera
) {
  const hw = layout.worldHalfW / cam.zoom
  const hh = layout.worldHalfH / cam.zoom
  const uNorm = (x - (cam.panX - hw)) / (2 * hw)
  // Match WebGL VS: tv = ((wpos.y - u_cam.y) + hh) / (2*hh)
  const vNorm = (y - cam.panY + hh) / (2 * hh)
  const sx = layout.viewX + uNorm * layout.viewW
  const sy = layout.viewY + vNorm * layout.viewH
  return [sx, sy] as const
}

/** Cap WebGL point sprites per frame (full N still simulated). */
const MAX_DRAW_PARTICLES = 200_000

function packInterleaved(sim: Sim): number {
  const N = Math.min(sim.spec.N, sim.x.length, sim.interleaved.length / 3)
  const u = sim.interleaved
  if (N <= MAX_DRAW_PARTICLES) {
    for (let i = 0; i < N; i++) {
      const o = i * 3
      u[o] = sim.x[i]
      u[o + 1] = sim.y[i]
      u[o + 2] = sim.type[i]
    }
    return N
  }
  const stride = Math.max(1, Math.ceil(N / MAX_DRAW_PARTICLES))
  const phase = sim.frame % stride
  let w = 0
  for (let i = phase; i < N; i += stride) {
    const o = w * 3
    u[o] = sim.x[i]
    u[o + 1] = sim.y[i]
    u[o + 2] = sim.type[i]
    w++
  }
  return w
}

function drawOverlay(
  sim: Sim,
  ctx: CanvasRenderingContext2D,
  layout: ViewLayout,
  cam: ViewCamera,
  gpuPhysics: boolean,
  particleDrawN: number,
  showHud: boolean
) {
  const { width, height, viewX, viewY, viewW, viewH } = layout
  ctx.clearRect(0, 0, width, height)
  if (!showHud) return

  const { showVel, showGrid, showTrailTiles } = sim.spec.overlays
  const N = Math.min(sim.spec.N, sim.x.length)
  const hw = layout.worldHalfW / cam.zoom
  const hh = layout.worldHalfH / cam.zoom
  const scaleXVis = viewW / (2 * hw)
  const scaleYVis = viewH / (2 * hh)

  if (showGrid) {
    ctx.save()
    ctx.beginPath()
    ctx.rect(viewX, viewY, viewW, viewH)
    ctx.clip()

    ctx.strokeStyle = "rgba(255,255,255,0.06)"
    ctx.lineWidth = 1
    const cell = sim.spec.cellSize
    const xStart = Math.floor((cam.panX - hw) / cell) * cell
    const yStart = Math.floor((cam.panY - hh) / cell) * cell
    for (let wx = xStart; wx <= cam.panX + hw + 1e-6; wx += cell) {
      const [sx0, sy0] = worldToScreen(wx, cam.panY - hh, layout, cam)
      const [sx1, sy1] = worldToScreen(wx, cam.panY + hh, layout, cam)
      ctx.beginPath()
      ctx.moveTo(sx0, sy0)
      ctx.lineTo(sx1, sy1)
      ctx.stroke()
    }
    for (let wy = yStart; wy <= cam.panY + hh + 1e-6; wy += cell) {
      const [sx0, sy0] = worldToScreen(cam.panX - hw, wy, layout, cam)
      const [sx1, sy1] = worldToScreen(cam.panX + hw, wy, layout, cam)
      ctx.beginPath()
      ctx.moveTo(sx0, sy0)
      ctx.lineTo(sx1, sy1)
      ctx.stroke()
    }
    ctx.restore()
  }

  if (showTrailTiles && !sim.spec.wrap) {
    const pointPxCss = 1.75
    const marginWorld = Math.max(
      0.06,
      (pointPxCss / Math.max(layout.viewW, 120)) * (2 * layout.worldHalfW)
    )
    const C = OPEN_TRAIL_CHUNK_WORLD
    const { ix0, ix1, iy0, iy1 } = openTrailVisibleChunkRange(
      cam,
      layout.worldHalfW,
      layout.worldHalfH,
      marginWorld
    )
    ctx.save()
    ctx.beginPath()
    ctx.rect(viewX, viewY, viewW, viewH)
    ctx.clip()
    ctx.strokeStyle = "rgba(0, 220, 180, 0.55)"
    ctx.lineWidth = 1
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
    ctx.fillStyle = "rgba(0, 220, 180, 0.7)"
    for (let iy = iy0; iy <= iy1; iy++) {
      for (let ix = ix0; ix <= ix1; ix++) {
        const wx0 = ix * C
        const wy0 = iy * C
        const wx1 = (ix + 1) * C
        const wy1 = (iy + 1) * C
        const [sx0, sy0] = worldToScreen(wx0, wy0, layout, cam)
        const [sx1, sy1] = worldToScreen(wx1, wy0, layout, cam)
        const [sx2, sy2] = worldToScreen(wx1, wy1, layout, cam)
        const [sx3, sy3] = worldToScreen(wx0, wy1, layout, cam)
        ctx.beginPath()
        ctx.moveTo(sx0, sy0)
        ctx.lineTo(sx1, sy1)
        ctx.lineTo(sx2, sy2)
        ctx.lineTo(sx3, sy3)
        ctx.closePath()
        ctx.stroke()
        const [tlx, tly] = worldToScreen(wx0 + C * 0.02, wy0 + C * 0.02, layout, cam)
        ctx.fillText(`${ix},${iy}`, tlx, tly)
      }
    }
    ctx.restore()
  }

  if (showVel && N > 0 && !gpuPhysics) {
    const maxVelDraw = 80_000
    const step = Math.max(1, Math.ceil(N / maxVelDraw))
    ctx.strokeStyle = "rgba(255,255,255,0.35)"
    ctx.lineWidth = 1
    for (let i = 0; i < N; i += step) {
      const [px, py] = worldToScreen(sim.x[i], sim.y[i], layout, cam)
      ctx.beginPath()
      ctx.moveTo(px, py)
      ctx.lineTo(
        px + sim.vx[i] * scaleXVis * 0.15,
        py + sim.vy[i] * scaleYVis * 0.15
      )
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
    `N=${sim.spec.N}  sprites=${particleDrawN}  K=${sim.K}  frame=${sim.frame}  WebGL${
      gpuPhysics ? " · WebGPU physics" : " · CPU physics"
    }`,
    `dt=${sim.spec.dt.toFixed(3)}  time=${physicsStepsFromSpec(sim.spec)}×  vMax=${sim.spec.vMax.toFixed(
      2
    )}  friction=${sim.spec.friction.toFixed(2)}  wrap=${sim.spec.wrap ? 1 : 0}`,
    `r̂Min=${sim.spec.rMin.toFixed(2)}  R̂=${sim.spec.R.toFixed(
      2
    )}  Rmax=${maxRMx(sim.RMx).toFixed(2)}  cell=${sim.spec.cellSize.toFixed(3)}`,
    `mutual=${sim.spec.mutualOnly ? 1 : 0}  settle=${
      sim.spec.settleEnabled ? 1 : 0
    }  k=${sim.spec.settleK.toFixed(2)}  sR=${sim.spec.settleR.toFixed(2)}`,
    `A[0,*]=[${A0 ?? ""}]`,
    `view  zoom=${cam.zoom.toFixed(2)}×  pan=(${cam.panX.toFixed(2)}, ${cam.panY.toFixed(2)})`,
  ]
  let ty = viewY + 16
  for (const ln of lines) {
    ctx.fillText(ln, viewX + 8, ty)
    ty += 14
  }

  ctx.strokeStyle = "rgba(255,255,255,0.08)"
  ctx.strokeRect(viewX + 0.5, viewY + 0.5, viewW - 1, viewH - 1)
}

// ========================= Matrix Toolbar =========================
const RADIUS_STEP_MIN = 0.002
const RADIUS_STEP_MAX = 0.005
const RADIUS_PAIR_EPS = 0.002

/**
 * Editable radius bounds in the matrix panel (clamp + randomize).
 * Keep MATRIX_RMAX_HI modest vs world vertical span (2): large R balloons spatial-hash
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

/** Seeded so first load is stable; bump seed for a different default rule set. */
const DEFAULT_RULES_BUILD_SEED = 9001

const DEFAULT_SPEC_MATRICES = (() => {
  const rng = mulberry32(DEFAULT_RULES_BUILD_SEED)
  const A = genRandomMatrix(DEFAULT_K, rng)
  const { rMinMx, RMx } = genRandomRMinTabMatrices(
    DEFAULT_K,
    rng,
    DEFAULT_R_MIN,
    DEFAULT_R_MAX
  )
  return { A, rMinMx, RMx }
})()

/** Default spec: toolbar defaults + random interaction matrices. */
const SPEC: Spec = {
  N: 2_500,
  K: DEFAULT_K,
  seed: 1337, // overwritten on load by random `seed` state; kept for saved-spec shape

  A: cloneMx(DEFAULT_SPEC_MATRICES.A),
  rMinMx: cloneMx(DEFAULT_SPEC_MATRICES.rMinMx),
  RMx: cloneMx(DEFAULT_SPEC_MATRICES.RMx),
  rMin: DEFAULT_R_MIN,
  R: DEFAULT_R_MAX,

  dt: 1 / 60,
  friction: 0.02,
  repel: DEFAULT_REPEL,
  forceFactor: DEFAULT_FORCE_FACTOR,
  vMax: 4,

  wrap: false,
  cellSize: 0.05,

  pixelScale: 800,
  genMatrix: false,

  overlays: {
    showVel: false,
    showGrid: false,
    showTrails: true,
    showTrailTiles: false,
  },
  trailPersistence: 0.96,

  mutualOnly: false,
  settleEnabled: false,
  settleK: 0.01,
  settleR: 0.08,
  timeScale: 1,
}

/** Integer physics substeps per render frame; clamped 1–TIME_SCALE_MAX. */
const TIME_SCALE_MIN = 1
const TIME_SCALE_MAX = 50

function physicsStepsFromSpec(sp: Spec): number {
  return clamp(
    Math.round(sp.timeScale),
    TIME_SCALE_MIN,
    TIME_SCALE_MAX
  )
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
            const nextA = genRandomMatrix(K, rng)
            const { rMinMx: nr, RMx: nR } = genRandomRMinTabMatrices(
              K,
              rng,
              scalarRMin,
              scalarR
            )
            onChangeA(nextA)
            onApplyRadii(nr, nR)
          }}
          title="Randomize forces, min r, and max r matrices"
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
/** WebGL canvas + transparent 2D overlay; full-area viewport in shared `layoutRef`. */
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

/** New placement + rules salt on each full page load (React state seed is random). */
function randomParticleSeed(): number {
  return Math.floor(Math.random() * 1e9)
}

// ========================= Component =========================
export default function App() {
  const debugHud = useMemo(
    () =>
      typeof window !== "undefined" &&
      new URLSearchParams(window.location.search).get("debug") === "1",
    []
  )

  // Use SPEC directly; do not override A on first render.
  const [spec, setSpec] = useState<Spec>({ ...SPEC })
  const [seed, setSeed] = useState<number>(() => randomParticleSeed())
  const [paused, setPaused] = useState(false)
  const [showRules, setShowRules] = useState(false)
  const [fps, setFps] = useState(0)

  const simRef = useRef<Sim | null>(null)
  /** Bumps when sim buffer is recreated (reset / spec reinit) so trail state can clear. */
  const simTrailEpochRef = useRef(0)
  /** Tracks seed seen by physics reinit so we reroll interaction rules on seed change, not on dt/friction tweaks. */
  const lastInitSeedRef = useRef<number | undefined>(undefined)
  const gpuRunnerRef = useRef<GpuSimRunner | null>(null)
  const viewCameraRef = useRef<ViewCamera>(defaultViewCamera())
  const [gpuPhysics, setGpuPhysics] = useState(false)

  // -------- Organism tracker (Competitors panel)
  const trackerRef = useRef<OrganismTracker | null>(null)
  if (trackerRef.current === null) {
    trackerRef.current = new OrganismTracker(TYPE_COLORS)
  }
  const [trackerSnapshot, setTrackerSnapshot] = useState<TrackerSnapshot>(
    () => ({ alive: [], dead: [], frameAt: 0, highlightId: null })
  )
  const [organismHighlight, setOrganismHighlight] = useState<number | null>(
    null
  )
  const [competitorsCollapsed, setCompetitorsCollapsed] = useState(false)
  const competitorsCollapsedRef = useRef(competitorsCollapsed)
  useEffect(() => {
    competitorsCollapsedRef.current = competitorsCollapsed
  }, [competitorsCollapsed])

  const {
    glCanvasRef,
    overlayCanvasRef,
    containerRef,
    layoutRef,
    particleGLRef,
    overlayCtxRef,
  } = useRendererViewport(simRef)

  // Keep `spec.seed` aligned with sim placement seed (footer, exports, saves).
  useEffect(() => {
    setSpec((s) => (s.seed === seed ? s : { ...s, seed }))
  }, [seed])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const viewport = el
    const cam = viewCameraRef
    const layoutRefLocal = layoutRef
    /** Active pointers (needed for pinch; single-touch uses same map). */
    const pointers = new Map<number, { x: number; y: number }>()
    let dragging = false
    let lx = 0
    let ly = 0
    /** World point + zoom anchored at pinch start (two fingers). */
    let pinch:
      | {
          dist0: number
          zoom0: number
          worldX: number
          worldY: number
        }
      | undefined

    function worldUnderClient(clientX: number, clientY: number): {
      worldX: number
      worldY: number
      nx: number
      ny: number
      layout: ViewLayout
    } | null {
      const layout = layoutRefLocal.current
      if (!layout) return null
      const rect = viewport.getBoundingClientRect()
      const mx = clientX - rect.left
      const my = clientY - rect.top
      const nx = (mx - layout.viewX) / layout.viewW
      const ny = (my - layout.viewY) / layout.viewH
      const { panX, panY, zoom } = cam.current
      const hw = layout.worldHalfW / zoom
      const hh = layout.worldHalfH / zoom
      const worldX = panX + (nx - 0.5) * 2 * hw
      const worldY = panY + (ny - 0.5) * 2 * hh
      return { worldX, worldY, nx, ny, layout }
    }

    function initPinchFromTwoTouches() {
      const ids = [...pointers.keys()]
      if (ids.length < 2) return
      const pa = pointers.get(ids[0]!)!
      const pb = pointers.get(ids[1]!)!
      const cx = (pa.x + pb.x) * 0.5
      const cy = (pa.y + pb.y) * 0.5
      const dist = Math.hypot(pb.x - pa.x, pb.y - pa.y)
      const w = worldUnderClient(cx, cy)
      if (!w) return
      pinch = {
        dist0: Math.max(dist, 1e-6),
        zoom0: cam.current.zoom,
        worldX: w.worldX,
        worldY: w.worldY,
      }
    }

    const onPointerDown = (e: PointerEvent) => {
      if (e.pointerType === "mouse" && e.button !== 0) return
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })
      el.style.cursor = "grabbing"
      if (pointers.size === 1) {
        dragging = true
        pinch = undefined
        lx = e.clientX
        ly = e.clientY
        try {
          el.setPointerCapture(e.pointerId)
        } catch {
          /* ignore */
        }
      } else if (pointers.size === 2) {
        dragging = false
        const firstId = [...pointers.keys()][0]!
        try {
          el.releasePointerCapture(firstId)
        } catch {
          /* ignore */
        }
        initPinchFromTwoTouches()
      }
    }
    const onPointerMove = (e: PointerEvent) => {
      if (!pointers.has(e.pointerId)) return
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })
      const layout = layoutRefLocal.current
      if (!layout) return

      if (pointers.size >= 2) {
        const ids = [...pointers.keys()]
        const pa = pointers.get(ids[0]!)!
        const pb = pointers.get(ids[1]!)!
        const cx = (pa.x + pb.x) * 0.5
        const cy = (pa.y + pb.y) * 0.5
        const dist = Math.hypot(pb.x - pa.x, pb.y - pa.y)
        if (!pinch) {
          initPinchFromTwoTouches()
        }
        const st = pinch
        if (st) {
          const newZoom = Math.min(
            ZOOM_MAX,
            Math.max(ZOOM_MIN, st.zoom0 * (dist / st.dist0))
          )
          const rect = viewport.getBoundingClientRect()
          const mx = cx - rect.left
          const my = cy - rect.top
          const nx = (mx - layout.viewX) / layout.viewW
          const ny = (my - layout.viewY) / layout.viewH
          const hw2 = layout.worldHalfW / newZoom
          const hh2 = layout.worldHalfH / newZoom
          cam.current.zoom = newZoom
          cam.current.panX = st.worldX - (nx - 0.5) * 2 * hw2
          cam.current.panY = st.worldY - (ny - 0.5) * 2 * hh2
        }
        return
      }

      if (!dragging) return
      const dx = e.clientX - lx
      const dy = e.clientY - ly
      lx = e.clientX
      ly = e.clientY
      const z = cam.current.zoom
      const hw = layout.worldHalfW / z
      const hh = layout.worldHalfH / z
      cam.current.panX -= (dx / layout.viewW) * 2 * hw
      cam.current.panY -= (dy / layout.viewH) * 2 * hh
    }
    const endDrag = (e?: PointerEvent) => {
      if (e) {
        try {
          el.releasePointerCapture(e.pointerId)
        } catch {
          /* ignore */
        }
      }
    }
    const onPointerUp = (e: PointerEvent) => {
      endDrag(e)
      pointers.delete(e.pointerId)
      if (pointers.size === 0) {
        dragging = false
        pinch = undefined
        el.style.cursor = "grab"
      } else if (pointers.size === 1) {
        pinch = undefined
        const rem = [...pointers.values()][0]!
        lx = rem.x
        ly = rem.y
        dragging = true
        const pid = [...pointers.keys()][0]!
        try {
          el.setPointerCapture(pid)
        } catch {
          /* ignore */
        }
      }
    }
    const onPointerCancel = (e: PointerEvent) => {
      pointers.delete(e.pointerId)
      if (pointers.size === 0) {
        dragging = false
        pinch = undefined
        el.style.cursor = "grab"
      } else if (pointers.size === 1) {
        pinch = undefined
        const rem = [...pointers.values()][0]!
        lx = rem.x
        ly = rem.y
        dragging = true
      }
    }
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const layout = layoutRefLocal.current
      if (!layout) return
      const rect = viewport.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      const { panX, panY, zoom } = cam.current
      const hw = layout.worldHalfW / zoom
      const hh = layout.worldHalfH / zoom
      const nx = (mx - layout.viewX) / layout.viewW
      const ny = (my - layout.viewY) / layout.viewH
      const worldX = panX + (nx - 0.5) * 2 * hw
      const worldY = panY + (ny - 0.5) * 2 * hh
      const factor = Math.exp(-e.deltaY * 0.0015)
      const newZoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoom * factor))
      const hw2 = layout.worldHalfW / newZoom
      const hh2 = layout.worldHalfH / newZoom
      cam.current.panX = worldX - (nx - 0.5) * 2 * hw2
      cam.current.panY = worldY - (ny - 0.5) * 2 * hh2
      cam.current.zoom = newZoom
    }
    const onDblClick = () => {
      cam.current = defaultViewCamera()
    }

    el.style.cursor = "grab"
    el.style.touchAction = "none"
    el.addEventListener("pointerdown", onPointerDown)
    el.addEventListener("pointermove", onPointerMove)
    el.addEventListener("pointerup", onPointerUp)
    el.addEventListener("pointercancel", onPointerCancel)
    el.addEventListener("lostpointercapture", onPointerCancel)
    el.addEventListener("wheel", onWheel, { passive: false })
    el.addEventListener("dblclick", onDblClick)
    return () => {
      el.removeEventListener("pointerdown", onPointerDown)
      el.removeEventListener("pointermove", onPointerMove)
      el.removeEventListener("pointerup", onPointerUp)
      el.removeEventListener("pointercancel", onPointerCancel)
      el.removeEventListener("lostpointercapture", onPointerCancel)
      el.removeEventListener("wheel", onWheel)
      el.removeEventListener("dblclick", onDblClick)
    }
  }, [containerRef, layoutRef])

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

    const rng = mulberry32(Date.now())
    const A = genRandomMatrix(spec.K, rng)
    const { rMinMx, RMx } = genRandomRMinTabMatrices(
      spec.K,
      rng,
      spec.rMin,
      spec.R
    )
    setSpec((s) => ({
      ...s,
      A,
      rMinMx,
      RMx,
      genMatrix: false,
    }))
  }, [spec.K, spec.genMatrix])

  // -------- Initialize / Re-initialize simulation when core layout changes.
  useEffect(() => {
    const firstInit = lastInitSeedRef.current === undefined
    const seedChanged = lastInitSeedRef.current !== seed
    lastInitSeedRef.current = seed
    const rerollInteractionRules = firstInit || seedChanged
    const { sim, specPatch } = initSim(spec, seed, {
      rerollInteractionRules,
    })
    simRef.current = sim
    if (specPatch) setSpec((s) => ({ ...s, ...specPatch }))
    simTrailEpochRef.current++
    viewCameraRef.current = defaultViewCamera()
    gpuRunnerRef.current?.uploadParticleState(simRef.current)
    trackerRef.current?.reset()
    setTrackerSnapshot({ alive: [], dead: [], frameAt: 0, highlightId: null })
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

  // -------- WebGPU physics: toroidal domain only (GPU matches fixed world_half bounce/wrap).
  useEffect(() => {
    let cancelled = false
    setGpuPhysics(false)
    gpuRunnerRef.current?.dispose()
    gpuRunnerRef.current = null
    if (!ENABLE_WEBGPU_PHYSICS || !spec.wrap) return

    void (async () => {
      const runner = await createGpuSimRunner(
        MAX_PARTICLES + 50_000,
        maxGridCellsForCellSize(spec.cellSize)
      )
      if (cancelled) {
        runner?.dispose()
        return
      }
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
  }, [spec.cellSize, spec.wrap])

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

      syncSimWorldToLayout(sim, layout)

      if (!paused) {
        const k = physicsStepsFromSpec(sim.spec)
        const gpu = gpuRunnerRef.current
        const useGpu = gpu && gpuPhysics && sim.spec.wrap
        if (useGpu) {
          gpu.subSteps = k
          gpu.step(sim)
          // GPU path doesn't bump sim.frame; CPU step() does.
          sim.frame++
        } else {
          for (let s = 0; s < k; s++) {
            step(sim)
          }
        }
      }

      const n = Math.min(sim.spec.N, sim.x.length)
      ensureParticleBufferCapacity(pr, n)
      const drawN = packInterleaved(sim)
      const stride =
        drawN > 0 && n > drawN ? Math.max(1, Math.ceil(n / drawN)) : 1
      const pointPxCss =
        stride > 2 ? 1.05 : stride > 1 ? 1.25 : n > 400_000 ? 1.2 : 1.75

      if (!paused) {
        trackerRef.current?.observe(
          sim,
          {
            layout: {
              width: layout.width,
              height: layout.height,
              dpr: layout.dpr,
              viewX: layout.viewX,
              viewY: layout.viewY,
              viewW: layout.viewW,
              viewH: layout.viewH,
              worldHalfW: layout.worldHalfW,
              worldHalfH: layout.worldHalfH,
            },
            camera: { ...viewCameraRef.current },
            pointPxCss,
          },
          { skipHudThumbnails: competitorsCollapsedRef.current }
        )
      }

      drawParticles(
        pr,
        sim.interleaved,
        drawN,
        layout,
        viewCameraRef.current,
        sim.K,
        TYPE_COLORS,
        pointPxCss,
        {
          trails: sim.spec.overlays.showTrails,
          trailPersistence: sim.spec.trailPersistence,
          tileWrap: sim.spec.wrap,
          trailHistKey: simTrailEpochRef.current,
        }
      )
      drawOverlay(
        sim,
        octx,
        layout,
        viewCameraRef.current,
        gpuPhysics && sim.spec.wrap,
        drawN,
        debugHud
      )

      raf = requestAnimationFrame(tick)
    }

    raf = requestAnimationFrame(tick)
    return () => {
      stopped = true
      cancelAnimationFrame(raf)
    }
  }, [paused, gpuPhysics, debugHud])

  // FPS meter (header only in debug HUD mode)
  useEffect(() => {
    if (!debugHud) return
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
  }, [debugHud])

  // Publish tracker snapshot to React — ~30fps when competitors panel shows live thumbs,
  // slower when collapsed to reduce main-thread wakeups.
  const organismHighlightRef = useRef<number | null>(null)
  useEffect(() => {
    organismHighlightRef.current = organismHighlight
  }, [organismHighlight])
  useEffect(() => {
    let cancelled = false
    let id: ReturnType<typeof setTimeout> | null = null
    const tick = () => {
      if (cancelled) return
      const tracker = trackerRef.current
      if (tracker) {
        setTrackerSnapshot({
          ...tracker.snapshot(organismHighlightRef.current),
          thumbnailDpr: layoutRef.current?.dpr ?? 1,
          thumbnailViewAspect: layoutRef.current
            ? layoutRef.current.viewW / layoutRef.current.viewH
            : undefined,
        })
      }
      const delay = competitorsCollapsedRef.current ? 250 : 1000 / 30
      id = setTimeout(tick, delay)
    }
    tick()
    return () => {
      cancelled = true
      if (id !== null) clearTimeout(id)
    }
  }, [competitorsCollapsed])

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
    const { sim, specPatch } = initSim(spec, seed, {
      rerollInteractionRules: true,
    })
    simRef.current = sim
    if (specPatch) setSpec((s) => ({ ...s, ...specPatch }))
    simTrailEpochRef.current++
    viewCameraRef.current = defaultViewCamera()
    gpuRunnerRef.current?.uploadParticleState(simRef.current)
    trackerRef.current?.reset()
    setTrackerSnapshot({ alive: [], dead: [], frameAt: 0, highlightId: null })
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
        gridTemplateRows: debugHud ? "auto 1fr auto" : "1fr",
        height: "100%",
        overflow: "hidden",
        background: "#0a0a0a",
        color: "#eaeaea",
        fontFamily:
          "ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial",
      }}
    >
      {/* Header (debug only) */}
      {debugHud && (
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
              title="Fading motion trails (GPU: previous frame retained with fade)"
            >
              <input
                type="checkbox"
                checked={spec.overlays.showTrails}
                onChange={(e) =>
                  setSpec({
                    ...spec,
                    overlays: { ...spec.overlays, showTrails: e.target.checked },
                  })
                }
              />
              Trails
            </label>
            <label
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 4,
                opacity: spec.wrap ? 0.5 : 1,
              }}
              title={
                spec.wrap
                  ? "Trail chunk tiles apply to open world only (disable Wrap)"
                  : "Show trail buffer chunk boundaries (open-world GPU tiles)"
              }
            >
              <input
                type="checkbox"
                disabled={spec.wrap}
                checked={!!spec.overlays.showTrailTiles}
                onChange={(e) =>
                  setSpec({
                    ...spec,
                    overlays: {
                      ...spec.overlays,
                      showTrailTiles: e.target.checked,
                    },
                  })
                }
              />
              Trail tiles
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
              <span
                style={{ opacity: 0.7 }}
                title="Physics steps per frame (simulation time speed)"
              >
                time:
              </span>
              <button
                type="button"
                onClick={() =>
                  setSpec((s) => ({
                    ...s,
                    timeScale: inc(s.timeScale, -5, TIME_SCALE_MIN, TIME_SCALE_MAX),
                  }))
                }
                style={chipStyle}
                title="−5×"
              >
                −5
              </button>
              <button
                type="button"
                onClick={() =>
                  setSpec((s) => ({
                    ...s,
                    timeScale: inc(s.timeScale, -1, TIME_SCALE_MIN, TIME_SCALE_MAX),
                  }))
                }
                style={chipStyle}
                title="−1×"
              >
                −
              </button>
              <span>{physicsStepsFromSpec(spec)}×</span>
              <button
                type="button"
                onClick={() =>
                  setSpec((s) => ({
                    ...s,
                    timeScale: inc(s.timeScale, +1, TIME_SCALE_MIN, TIME_SCALE_MAX),
                  }))
                }
                style={chipStyle}
                title="+1×"
              >
                +
              </button>
              <button
                type="button"
                onClick={() =>
                  setSpec((s) => ({
                    ...s,
                    timeScale: inc(s.timeScale, +5, TIME_SCALE_MIN, TIME_SCALE_MAX),
                  }))
                }
                style={chipStyle}
                title="+5×"
              >
                +5
              </button>
            </div>

            <div
              style={{ display: "inline-flex", gap: 6, alignItems: "center" }}
            >
              <span
                style={{ opacity: 0.7 }}
                title="Trail persistence per frame (higher = longer fading tail)"
              >
                trail:
              </span>
              <button
                type="button"
                onClick={() =>
                  setSpec((s) => ({
                    ...s,
                    trailPersistence: inc(s.trailPersistence, -0.02, 0.75, 0.98),
                  }))
                }
                style={chipStyle}
              >
                −
              </button>
              <span>{spec.trailPersistence.toFixed(2)}</span>
              <button
                type="button"
                onClick={() =>
                  setSpec((s) => ({
                    ...s,
                    trailPersistence: inc(s.trailPersistence, +0.02, 0.75, 0.98),
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
      )}

      {/* Canvas area (middle row) */}
      <div
        ref={containerRef}
        style={{
          position: "relative",
          width: "100%",
          height: "100%",
          minHeight: 0,
          overflow: "hidden",
          userSelect: "none",
          WebkitUserSelect: "none",
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
        <Competitors
          snapshot={trackerSnapshot}
          stabilityGate={trackerRef.current?.params.stabilityGate ?? 0.95}
          symmetryGate={trackerRef.current?.params.symmetryGate ?? 0.9}
          leaderboardGraceSeconds={
            trackerRef.current?.params.leaderboardGraceSeconds ?? 4
          }
          topN={trackerRef.current?.params.topN ?? 8}
          onHover={setOrganismHighlight}
          collapsed={competitorsCollapsed}
          onToggleCollapse={() => setCompetitorsCollapsed((v) => !v)}
        />
      </div>

      {/* Footer (debug only) */}
      {debugHud && (
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
          {spec.dt} · time={physicsStepsFromSpec(spec)}× · friction={spec.friction.toFixed(2)} · vMax={spec.vMax} ·
          cell={spec.cellSize} · seed={seed}
        </span>
        <span style={{ opacity: 0.65, marginLeft: 12 }}>
          Drag to pan · pinch or wheel zoom · double-click reset view
        </span>
      </div>
      )}

      {/* Rules toolbar */}
      {debugHud && showRules && (
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
