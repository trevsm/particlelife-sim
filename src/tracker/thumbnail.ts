/**
 * HUD thumbnails: fixed virtual zoom, CoM-centered; live ~30fps from current sim.
 * Particles draw as quantized `Pf` squares. Moving clusters get a short **motion
 * smear** (faded copies behind −v̂), not a long CoM tail.
 */

import type { Sim } from "../simTypes"
import type { ViewCamera, ViewLayout } from "../webglParticles"
import type { Organism } from "./types"

export type ThumbnailViewport = {
  layout: Pick<
    ViewLayout,
    | "width"
    | "height"
    | "dpr"
    | "viewX"
    | "viewY"
    | "viewW"
    | "viewH"
    | "worldHalfW"
    | "worldHalfH"
  >
  /** Ignored for rendering — thumbs use CoM-centered pan + `THUMB_VIRTUAL_ZOOM`. */
  camera: ViewCamera
  pointPxCss: number
}

/** Keep in sync with `App.tsx` `ZOOM_MIN` — virtual zoom for HUD thumbs (“all the way out”). */
const THUMB_VIRTUAL_ZOOM = 0.12

/** Short motion-smear behind −v̂ (world space per step), not a long CoM tail. */
const MOTION_BLUR_STEPS = 6

const THUMB_FALLBACK_CSS = 48
/** HUD cap on the longer framebuffer edge before uniform downscale */
const THUMB_MAX_EDGE_CSS = 56
function makeCanvas2D(w: number, h: number): {
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null
  getImageData: () => ImageData
} {
  const width = Math.max(1, Math.floor(w))
  const height = Math.max(1, Math.floor(h))
  if (typeof OffscreenCanvas !== "undefined") {
    const c = new OffscreenCanvas(width, height)
    const ctx = c.getContext("2d") as OffscreenCanvasRenderingContext2D | null
    return {
      ctx,
      getImageData: () =>
        ctx
          ? ctx.getImageData(0, 0, width, height)
          : new ImageData(width, height),
    }
  }
  if (typeof document === "undefined") {
    return {
      ctx: null,
      getImageData: () => new ImageData(width, height),
    }
  }
  const c = document.createElement("canvas")
  c.width = width
  c.height = height
  const ctx = c.getContext("2d")
  return {
    ctx,
    getImageData: () =>
      ctx ? ctx.getImageData(0, 0, width, height) : new ImageData(width, height),
  }
}

/** WebGL VS: snapped framebuffer centers in device px. */
function worldToFramebufferSnappedXY(
  wx: number,
  wy: number,
  layout: ThumbnailViewport["layout"],
  cam: ViewCamera
): { fx: number; fy: number } {
  const invz = 1 / Math.max(cam.zoom, 1e-6)
  const hw = layout.worldHalfW * invz
  const hh = layout.worldHalfH * invz
  const tu = ((wx - cam.panX) + hw) / (2 * hw)
  const tv = ((wy - cam.panY) + hh) / (2 * hh)
  const screenX = layout.viewX + tu * layout.viewW
  const screenY = layout.viewY + tv * layout.viewH
  const fx = Math.floor((screenX / layout.width) * (layout.width * layout.dpr)) + 0.5
  const fy = Math.floor((screenY / layout.height) * (layout.height * layout.dpr)) + 0.5
  return { fx, fy }
}

function glPointDiameter(vp: ThumbnailViewport): number {
  const dpr = Math.min(4, Math.max(1, vp.layout.dpr ?? 1))
  const z = Math.max(vp.camera.zoom, 1e-6)
  return Math.max(1, Math.floor(vp.pointPxCss * dpr * z + 0.5))
}

function particleScreenRect(vp: ThumbnailViewport, Pf: number, wx: number, wy: number) {
  const { fx, fy } = worldToFramebufferSnappedXY(wx, wy, vp.layout, vp.camera)
  const half = (Pf - 1) * 0.5
  const left = Math.floor(fx - half)
  const top = Math.floor(fy - half)
  return { left, top }
}

function unwrapTowardOrg(
  x: number,
  y: number,
  cx: number,
  cy: number,
  wrap: boolean,
  whw: number,
  whh: number,
  worldW: number,
  worldH: number
): [number, number] {
  let xi = x
  let yi = y
  if (wrap) {
    const dx = xi - cx
    if (dx > whw) xi -= worldW
    else if (dx < -whw) xi += worldW
    const dy = yi - cy
    if (dy > whh) yi -= worldH
    else if (dy < -whh) yi += worldH
  }
  return [xi, yi]
}

/** Solid fill from `#rrggbb` + alpha channel. */
function fillStyleWithAlpha(hex: string, alpha: number): string {
  const h = hex.trim()
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(h)
  if (!m)
    return `rgba(210,210,210,${Math.max(0, Math.min(1, alpha))})`
  return `rgba(${parseInt(m[1]!, 16)},${parseInt(m[2]!, 16)},${parseInt(
    m[3]!,
    16,
  )},${Math.max(0, Math.min(1, alpha))})`
}

/** Minimum crop with framebuffer aspect ratio bufW:bufH, centered on bbox. */
function cropFrustumWithAspect(
  bufW: number,
  bufH: number,
  minLeft: number,
  minTop: number,
  maxRight: number,
  maxBottom: number,
  marginPx: number
): { cropLeft: number; cropTop: number; cropW: number; cropH: number } {
  const minL = minLeft - marginPx
  const minT = minTop - marginPx
  const maxR = maxRight + marginPx
  const maxB = maxBottom + marginPx

  const w = Math.max(1, maxR - minL + 1)
  const h = Math.max(1, maxB - minT + 1)
  const aspect = bufW / bufH

  let cropW = w
  let cropH = h
  if (w / h > aspect) {
    cropH = w / aspect
    cropW = w
  } else {
    cropW = h * aspect
    cropH = h
  }

  const cx = (minL + maxR) * 0.5
  const cy = (minT + maxB) * 0.5
  const cropLeft = cx - cropW * 0.5
  const cropTop = cy - cropH * 0.5

  return { cropLeft, cropTop, cropW, cropH }
}

/** Fallback (no viewport): square fit bbox in THUMB_FALLBACK_CSS. */
function renderFallbackBBox(
  org: Organism,
  sim: Sim,
  colors: string[]
): ImageData {
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
  const Pf =
    org.members.size > 80 ? 1 : org.members.size > 24 ? 2 : 3
  const halfDot = (Pf - 1) * 0.5

  const xs = new Float32Array(N)
  const ys = new Float32Array(N)
  const ts = new Uint16Array(N)
  let ki = 0
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
    xs[ki] = xi
    ys[ki] = yi
    ts[ki] = TYPE[idx] | 0
    if (xi < minX) minX = xi
    if (xi > maxX) maxX = xi
    if (yi < minY) minY = yi
    if (yi > maxY) maxY = yi
    ki++
  }
  const spanTight = Math.max(1e-6, Math.max(maxX - minX, maxY - minY))
  const spanLoose = spanTight * 1.18
  const margin = 4
  const drawSize = THUMB_FALLBACK_CSS - 2 * margin
  const scalePos = drawSize / spanLoose
  const half = THUMB_FALLBACK_CSS * 0.5
  const midX = (minX + maxX) * 0.5
  const midY = (minY + maxY) * 0.5

  const { ctx, getImageData } = makeCanvas2D(THUMB_FALLBACK_CSS, THUMB_FALLBACK_CSS)
  if (!ctx) return new ImageData(THUMB_FALLBACK_CSS, THUMB_FALLBACK_CSS)
  ctx.imageSmoothingEnabled = false
  ctx.fillStyle = "#0a0a0a"
  ctx.fillRect(0, 0, THUMB_FALLBACK_CSS, THUMB_FALLBACK_CSS)
  for (let k = 0; k < N; k++) {
    const px = Math.floor(half + (xs[k] - midX) * scalePos - halfDot)
    const py = Math.floor(half + (ys[k] - midY) * scalePos - halfDot)
    ctx.fillStyle = colors[ts[k] % colors.length]
    ctx.fillRect(px, py, Pf, Pf)
  }
  return getImageData()
}

export function renderThumbnail(
  org: Organism,
  sim: Sim,
  colors: string[],
  viewport?: ThumbnailViewport
): ImageData {
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
  if (N === 0) return new ImageData(1, 1)

  const vp = viewport
  if (
    !vp ||
    vp.layout.width < 2 ||
    vp.layout.height < 2 ||
    vp.layout.worldHalfW <= 0 ||
    !Number.isFinite(vp.pointPxCss)
  ) {
    return renderFallbackBBox(org, sim, colors)
  }

  /** Virtual frustum: max zoom-out + look at organism CoM (matches main GL math, not user cam). */
  const thumbVp: ThumbnailViewport = {
    layout: vp.layout,
    camera: {
      panX: cx,
      panY: cy,
      zoom: THUMB_VIRTUAL_ZOOM,
    },
    pointPxCss: vp.pointPxCss,
  }

  const bufW = vp.layout.width * vp.layout.dpr
  const bufH = vp.layout.height * vp.layout.dpr
  const Pf = glPointDiameter(thumbVp)

  type Rect = { left: number; top: number; Pf: number; typeIdx: number }
  type MotionBlur = {
    left: number
    top: number
    pf: number
    typeIdx: number
    alpha: number
  }

  const rects: Rect[] = []
  const blurLayer: MotionBlur[] = []

  type PW = { wx: number; wy: number; typeIdx: number }
  const parts: PW[] = []

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
    parts.push({ wx: xi, wy: yi, typeIdx: TYPE[idx] | 0 })
    const { left, top } = particleScreenRect(thumbVp, Pf, xi, yi)
    rects.push({
      left,
      top,
      Pf,
      typeIdx: TYPE[idx] | 0,
    })
  }

  const vcx = org.vCom[0]
  const vcy = org.vCom[1]
  const speed = Math.hypot(vcx, vcy)
  if (speed > 1e-6) {
    const inv = 1 / speed
    const nx = vcx * inv
    const ny = vcy * inv
    const lag0 = Math.max(sim.worldHalfW * 0.011, org.rg * 0.62, 2.2e-3)
    const blurPf = Math.max(1, Math.floor(Pf * 0.96))
    for (const p of parts) {
      for (let L = 1; L <= MOTION_BLUR_STEPS; L++) {
        const [bx, by] = unwrapTowardOrg(
          p.wx - nx * lag0 * L,
          p.wy - ny * lag0 * L,
          cx,
          cy,
          wrap,
          whw,
          whh,
          worldW,
          worldH
        )
        const sr = particleScreenRect(thumbVp, blurPf, bx, by)
        const alpha = Math.min(0.52, 0.48 / L)
        blurLayer.push({
          left: sr.left,
          top: sr.top,
          pf: blurPf,
          typeIdx: p.typeIdx,
          alpha,
        })
      }
    }
  }

  let minLeft = Infinity
  let minTop = Infinity
  let maxRight = -Infinity
  let maxBottom = -Infinity
  for (const r of rects) {
    minLeft = Math.min(minLeft, r.left)
    minTop = Math.min(minTop, r.top)
    maxRight = Math.max(maxRight, r.left + r.Pf - 1)
    maxBottom = Math.max(maxBottom, r.top + r.Pf - 1)
  }
  for (const b of blurLayer) {
    minLeft = Math.min(minLeft, b.left)
    minTop = Math.min(minTop, b.top)
    maxRight = Math.max(maxRight, b.left + b.pf - 1)
    maxBottom = Math.max(maxBottom, b.top + b.pf - 1)
  }

  const marginPx = Math.max(4, Pf + 2)
  const { cropLeft, cropTop, cropW, cropH } = cropFrustumWithAspect(
    bufW,
    bufH,
    minLeft,
    minTop,
    maxRight,
    maxBottom,
    marginPx
  )

  const maxEdgeDev = Math.round(THUMB_MAX_EDGE_CSS * vp.layout.dpr)
  const need = Math.max(cropW, cropH)
  const scaleDown = need <= maxEdgeDev ? 1 : maxEdgeDev / need

  const outW = Math.max(Pf, Math.floor(cropW * scaleDown))
  const outH = Math.max(Pf, Math.floor(cropH * scaleDown))

  const { ctx, getImageData } = makeCanvas2D(outW, outH)
  if (!ctx) return new ImageData(outW, outH)

  ctx.imageSmoothingEnabled = false
  ctx.fillStyle = "#0a0a0a"
  ctx.fillRect(0, 0, outW, outH)

  for (const b of blurLayer) {
    const tx = Math.floor((b.left - cropLeft) * scaleDown)
    const ty = Math.floor((b.top - cropTop) * scaleDown)
    const bd = Math.max(1, Math.round(b.pf * scaleDown))
    ctx.fillStyle = fillStyleWithAlpha(
      colors[b.typeIdx % colors.length],
      b.alpha
    )
    ctx.fillRect(tx, ty, bd, bd)
  }

  for (const r of rects) {
    const tx = Math.floor((r.left - cropLeft) * scaleDown)
    const ty = Math.floor((r.top - cropTop) * scaleDown)
    const dot = Math.max(1, Math.round(r.Pf * scaleDown))
    ctx.fillStyle = colors[r.typeIdx % colors.length]
    ctx.fillRect(tx, ty, dot, dot)
  }

  return getImageData()
}

/** @deprecated — thumbs are variable size; use ImageData dimensions + `thumbnailDpr` */
export const THUMBNAIL_SIZE = THUMB_FALLBACK_CSS
