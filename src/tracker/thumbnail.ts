/**
 * Render a 48×48 ImageData snapshot of an organism, centered on its CoM and
 * scaled to fit. Particles are drawn as hard-edge `fillRect` squares with
 * smoothing disabled so the thumbnail reads as pixel art.
 *
 * Dot size follows the same **world extent** as WebGL point sprites: the vertex
 * shader maps `pointPxCss` with `gl_PointSize = pointPxCss * dpr * zoom`; the
 * footprint in world units is `pointPxCss * 2 * worldHalfW / viewW` (zoom
 * cancels). Scaling the bbox to the thumbnail then yields the same pixelation
 * density as the center view (for a given viewport width and point multiplier).
 *
 * Cleared each call (no trail) — a thumbnail is identity, not motion.
 */

import type { Sim } from "../simTypes"
import type { Organism } from "./types"

const THUMB_SIZE = 48

/** Must match the values used for `drawParticles(..., pointPxCss, ...)` in App. */
export type ThumbnailViewport = {
  worldHalfW: number
  /** layout.viewW — CSS viewport width used for world projection */
  viewW: number
  pointPxCss: number
}

type AnyCanvasCtx =
  | CanvasRenderingContext2D
  | OffscreenCanvasRenderingContext2D

function makeContext(): {
  ctx: AnyCanvasCtx | null
  getImage: () => ImageData
} {
  if (typeof OffscreenCanvas !== "undefined") {
    const c = new OffscreenCanvas(THUMB_SIZE, THUMB_SIZE)
    const ctx = c.getContext("2d") as OffscreenCanvasRenderingContext2D | null
    return {
      ctx,
      getImage: () =>
        ctx
          ? ctx.getImageData(0, 0, THUMB_SIZE, THUMB_SIZE)
          : new ImageData(THUMB_SIZE, THUMB_SIZE),
    }
  }
  if (typeof document === "undefined") {
    return { ctx: null, getImage: () => new ImageData(THUMB_SIZE, THUMB_SIZE) }
  }
  const c = document.createElement("canvas")
  c.width = THUMB_SIZE
  c.height = THUMB_SIZE
  const ctx = c.getContext("2d")
  return {
    ctx,
    getImage: () =>
      ctx
        ? ctx.getImageData(0, 0, THUMB_SIZE, THUMB_SIZE)
        : new ImageData(THUMB_SIZE, THUMB_SIZE),
  }
}

export function renderThumbnail(
  org: Organism,
  sim: Sim,
  colors: string[],
  viewport?: ThumbnailViewport
): ImageData {
  const { ctx, getImage } = makeContext()
  if (!ctx) return new ImageData(THUMB_SIZE, THUMB_SIZE)

  // Hard-pixel aesthetic — no smoothing on rect fills.
  ctx.imageSmoothingEnabled = false

  ctx.fillStyle = "#0a0a0a"
  ctx.fillRect(0, 0, THUMB_SIZE, THUMB_SIZE)

  const X = sim.x
  const Y = sim.y
  const TYPE = sim.type
  const wrap = sim.spec.wrap
  const whw = sim.worldHalfW
  const whh = sim.worldHalfH
  const worldW = 2 * whw
  const worldH = 2 * whh
  const cx = org.com[0]
  const cy = org.com[1]

  const N = org.members.size
  if (N === 0) return getImage()

  const xs = new Float32Array(N)
  const ys = new Float32Array(N)
  const ts = new Uint16Array(N)
  let i = 0
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  for (const idx of org.members) {
    let xi = X[idx]
    let yi = Y[idx]
    if (wrap) {
      const dx = xi - cx
      if (dx > whw) xi -= worldW
      else if (dx < -whw) xi += worldW
      const dy = yi - cy
      if (dy > whh) yi -= worldH
      else if (dy < -whh) yi += worldH
    }
    xs[i] = xi
    ys[i] = yi
    ts[i] = TYPE[idx] | 0
    if (xi < minX) minX = xi
    if (xi > maxX) maxX = xi
    if (yi < minY) minY = yi
    if (yi > maxY) maxY = yi
    i++
  }

  const spanX = Math.max(1e-6, maxX - minX)
  const spanY = Math.max(1e-6, maxY - minY)
  const span = Math.max(spanX, spanY) * 1.18
  const margin = 4
  const drawSize = THUMB_SIZE - 2 * margin
  const scale = drawSize / span
  const half = THUMB_SIZE * 0.5
  const midX = (minX + maxX) * 0.5
  const midY = (minY + maxY) * 0.5

  /** World-units diameter of a point sprite (∝ pointPxCss); same mapping as webglParticles VS. */
  let dotPx = 2
  const vp = viewport
  if (
    vp &&
    vp.viewW > 1 &&
    vp.worldHalfW > 0 &&
    Number.isFinite(vp.pointPxCss)
  ) {
    const particleWorldDiameter =
      (vp.pointPxCss * 2 * vp.worldHalfW) / vp.viewW
    const raw = particleWorldDiameter * scale
    dotPx = Math.max(1, Math.min(24, Math.floor(raw + 0.5)))
  } else {
    dotPx =
      org.members.size > 80 ? 1 : org.members.size > 24 ? 2 : 3
  }
  const half_dot = (dotPx - 1) * 0.5

  for (let k = 0; k < N; k++) {
    // Floor + integer offset → pixel-grid-aligned coordinates.
    const px = Math.floor(half + (xs[k] - midX) * scale - half_dot)
    const py = Math.floor(half + (ys[k] - midY) * scale - half_dot)
    ctx.fillStyle = colors[ts[k] % colors.length]
    ctx.fillRect(px, py, dotPx, dotPx)
  }

  return getImage()
}

export const THUMBNAIL_SIZE = THUMB_SIZE
