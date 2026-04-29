/** WebGL2 instanced-style point sprites for many particles (1M+ draw calls → one drawArrays). */

import { OpenTrailChunkGrid } from "./openTrailChunks"

/** Pan/zoom for mapping world → screen (zoom 1 = full world fits view). */
export type ViewCamera = {
  panX: number
  panY: number
  zoom: number
}

export const defaultViewCamera = (): ViewCamera => ({
  panX: 0,
  panY: 0,
  /** <1 zooms out (more world visible); 1 fits fundamental cell to view. */
  zoom: 0.85,
})

export type ViewLayout = {
  width: number
  height: number
  dpr: number
  viewX: number
  viewY: number
  /** Viewport width/height in CSS px (fills container; world maps with aspect). */
  viewW: number
  viewH: number
  /** World half-extents matching physics (horizontal follows view aspect). */
  worldHalfW: number
  worldHalfH: number
  /** CSS px per world unit (horizontal / vertical) for overlays. */
  scaleX: number
  scaleY: number
}

type TrailBundle = {
  fbos: [WebGLFramebuffer, WebGLFramebuffer]
  texs: [WebGLTexture, WebGLTexture]
  ping: 0 | 1
  w: number
  h: number
  fadeProgram: WebGLProgram
  compositeProgram: WebGLProgram
  emptyVao: WebGLVertexArrayObject
  uFadePrev: WebGLUniformLocation
  uFadeBg: WebGLUniformLocation
  uFadeKeep: WebGLUniformLocation
  uCompositeTex: WebGLUniformLocation
  uCompositeView: WebGLUniformLocation
  uCompositeWorldHalf: WebGLUniformLocation
  uCompositeCam: WebGLUniformLocation
  uCompositeTrailHalf: WebGLUniformLocation
  uCompositeBg: WebGLUniformLocation
  uCompositeFoldTorus: WebGLUniformLocation
  uCompositeTorusPeriod: WebGLUniformLocation
}

export type ParticleGL = {
  gl: WebGL2RenderingContext
  program: WebGLProgram
  /** Screen pass uses main VS; trail ping-pong uses world-fixed orthographic VS. */
  trailAccumProgram: WebGLProgram
  vao: WebGLVertexArrayObject
  buffer: WebGLBuffer
  locs: {
    aPosition: number
    aType: number
    uCanvas: WebGLUniformLocation
    uView: WebGLUniformLocation
    uWorldHalf: WebGLUniformLocation
    uCam: WebGLUniformLocation
    uTileShift: WebGLUniformLocation
    uPointPx: WebGLUniformLocation
    uPalette: WebGLUniformLocation
  }
  locsAccum: {
    uTileShift: WebGLUniformLocation
    uTrailHalf: WebGLUniformLocation
    uPointPx: WebGLUniformLocation
    uDpr: WebGLUniformLocation
    uWorldHalfRef: WebGLUniformLocation
    uDrawBuf: WebGLUniformLocation
    uPalette: WebGLUniformLocation
  }
  maxN: number
  paletteTex: WebGLTexture
  paletteK: number
  /** Ping-pong trail targets; lazily allocated. */
  trail: TrailBundle | undefined
  prevTileWrap: boolean | undefined
  trailHistKey: number | null
  /** Infinite open-world trail tiles (Your World of Text–style chunks); torus uses `trail`. */
  openTrailGrid: OpenTrailChunkGrid | undefined
  /** Temp copy for per-chunk particle splats in open-world trails. */
  trailScratch: Float32Array
}

function compile(
  gl: WebGL2RenderingContext,
  type: number,
  src: string
): WebGLShader {
  const sh = gl.createShader(type)!
  gl.shaderSource(sh, src)
  gl.compileShader(sh)
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh) || "shader compile error"
    gl.deleteShader(sh)
    throw new Error(log)
  }
  return sh
}

function link(
  gl: WebGL2RenderingContext,
  vs: WebGLShader,
  fs: WebGLShader
): WebGLProgram {
  const prog = gl.createProgram()!
  gl.attachShader(prog, vs)
  gl.attachShader(prog, fs)
  gl.linkProgram(prog)
  gl.deleteShader(vs)
  gl.deleteShader(fs)
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog) || "program link error"
    gl.deleteProgram(prog)
    throw new Error(log)
  }
  return prog
}

const VS = `#version 300 es
layout(location = 0) in vec2 a_position;
layout(location = 1) in float a_type;
out float v_type;
uniform vec3 u_canvas; // css width, css height, dpr
uniform vec4 u_view;   // viewX, viewY, viewW, viewH (css px)
uniform vec2 u_world_half; // physics torus half-extents
uniform vec3 u_cam;      // panX, panY, zoom (zoom >= 1 zooms in)
uniform vec2 u_tile_shift; // periodic copy offset (torus tiling); (0,0) for open world
uniform float u_pointPx;

void main() {
  v_type = a_type;
  vec2 wpos = a_position + u_tile_shift;
  float invz = 1.0 / max(u_cam.z, 1e-6);
  float hw = u_world_half.x * invz;
  float hh = u_world_half.y * invz;
  float tu = ((wpos.x - u_cam.x) + hw) / (2.0 * hw);
  float tv = ((wpos.y - u_cam.y) + hh) / (2.0 * hh);
  vec2 screen = vec2(
    u_view.x + tu * u_view.z,
    u_view.y + tv * u_view.w
  );
  // Pixel-perfect: snap to device-pixel centers (retro / crisp squares).
  float bufW = u_canvas.x * u_canvas.z;
  float bufH = u_canvas.y * u_canvas.z;
  float pxf = floor((screen.x / u_canvas.x) * bufW) + 0.5;
  float pyf = floor((screen.y / u_canvas.y) * bufH) + 0.5;
  vec2 ndc = vec2(
    (pxf / bufW) * 2.0 - 1.0,
    -((pyf / bufH) * 2.0 - 1.0)
  );
  gl_Position = vec4(ndc, 0.0, 1.0);
  float ps = u_pointPx * u_canvas.z * u_cam.z;
  gl_PointSize = max(1.0, floor(ps + 0.5));
}
`

export const PARTICLE_POINT_FS = `#version 300 es
precision highp float;
in float v_type;
uniform highp sampler2D u_palette;
out vec4 outColor;

void main() {
  int k = clamp(int(v_type + 0.5), 0, 255);
  vec3 col = texelFetch(u_palette, ivec2(k, 0), 0).rgb;
  // Solid square points (no circular discard) — less fragment work at huge N.
  outColor = vec4(col, 1.0);
}
`

const FS = PARTICLE_POINT_FS

/** Splat particles into trail FBO in world-fixed coordinates (no camera). */
const TRAIL_ACCUM_VS = `#version 300 es
layout(location = 0) in vec2 a_position;
layout(location = 1) in float a_type;
out float v_type;
uniform vec2 u_tile_shift;
uniform vec2 u_trail_half;
uniform float u_pointPx;
uniform float u_dpr;
uniform float u_world_half_ref;
uniform vec2 u_draw_buf;

void main() {
  v_type = a_type;
  vec2 wpos = a_position + u_tile_shift;
  float nx = wpos.x / u_trail_half.x;
  float ny = -wpos.y / u_trail_half.y;
  float pxf = floor((nx * 0.5 + 0.5) * u_draw_buf.x) + 0.5;
  float pyf = floor((ny * 0.5 + 0.5) * u_draw_buf.y) + 0.5;
  float snx = (pxf / u_draw_buf.x) * 2.0 - 1.0;
  float sny = (pyf / u_draw_buf.y) * 2.0 - 1.0;
  gl_Position = vec4(snx, sny, 0.0, 1.0);
  float scale = max(u_world_half_ref / u_trail_half.x, 0.05);
  float ps = u_pointPx * u_dpr * scale;
  gl_PointSize = max(1.0, floor(ps + 0.5));
}
`

const BG_RGB: [number, number, number] = [0.04, 0.04, 0.04]

const FULLSCREEN_VS = `#version 300 es
out vec2 v_uv;
void main() {
  vec2 p = vec2(
    gl_VertexID == 0 ? -1.0 : gl_VertexID == 1 ? 3.0 : -1.0,
    gl_VertexID == 0 ? -1.0 : gl_VertexID == 1 ? -1.0 : 3.0
  );
  gl_Position = vec4(p, 0.0, 1.0);
  v_uv = p * 0.5 + 0.5;
}
`

const TRAIL_FADE_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_prev;
uniform vec3 u_bg;
uniform float u_keep;
out vec4 outColor;

void main() {
  vec3 c = texture(u_prev, v_uv).rgb;
  outColor = vec4(mix(u_bg, c, u_keep), 1.0);
}
`

/** Map screen → world, sample world-aligned trail texture (pan/zoom only affect sampling). */
const TRAIL_COMPOSITE_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec4 u_view;
uniform vec2 u_world_half;
uniform vec3 u_cam;
uniform vec2 u_trail_half;
uniform vec3 u_bg;
uniform int u_fold_torus;
uniform vec2 u_torus_period;
out vec4 outColor;

void main() {
  vec2 screenCss = vec2(u_view.x + v_uv.x * u_view.z, u_view.y + v_uv.y * u_view.w);
  float tu = (screenCss.x - u_view.x) / u_view.z;
  float tv = (screenCss.y - u_view.y) / u_view.w;
  float invz = 1.0 / max(u_cam.z, 1e-6);
  float hw = u_world_half.x * invz;
  float hh = u_world_half.y * invz;
  float wx = (tu - 0.5) * 2.0 * hw + u_cam.x;
  float wy = (tv - 0.5) * 2.0 * hh + u_cam.y;
  if (u_fold_torus > 0) {
    wx = mod(wx + u_world_half.x, u_torus_period.x) - u_world_half.x;
    wy = mod(wy + u_world_half.y, u_torus_period.y) - u_world_half.y;
  }
  float uu = wx / u_trail_half.x * 0.5 + 0.5;
  float vv = (-wy / u_trail_half.y) * 0.5 + 0.5;
  if (uu < 0.0 || uu > 1.0 || vv < 0.0 || vv > 1.0) {
    outColor = vec4(u_bg, 1.0);
  } else {
    ivec2 ts = textureSize(u_tex, 0);
    int tx = clamp(int(floor(uu * float(ts.x))), 0, ts.x - 1);
    int ty = clamp(int(floor(vv * float(ts.y))), 0, ts.y - 1);
    outColor = vec4(texelFetch(u_tex, ivec2(tx, ty), 0).rgb, 1.0);
  }
}
`

/** Fixed trail world half-extent multiple for toroidal mode (wrap on). */
export const TRAIL_TORUS_HALF_MULT = 14

function destroyTrailBundle(gl: WebGL2RenderingContext, t: TrailBundle) {
  gl.deleteFramebuffer(t.fbos[0])
  gl.deleteFramebuffer(t.fbos[1])
  gl.deleteTexture(t.texs[0])
  gl.deleteTexture(t.texs[1])
  gl.deleteProgram(t.fadeProgram)
  gl.deleteProgram(t.compositeProgram)
  gl.deleteVertexArray(t.emptyVao)
}

function createTrailBundle(gl: WebGL2RenderingContext, w: number, h: number): TrailBundle {
  const makeTex = () => {
    const tex = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, tex)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
    gl.bindTexture(gl.TEXTURE_2D, null)
    return tex
  }

  const tex0 = makeTex()
  const tex1 = makeTex()
  const fbo0 = gl.createFramebuffer()!
  const fbo1 = gl.createFramebuffer()!

  for (let i = 0; i < 2; i++) {
    const fbo = i === 0 ? fbo0 : fbo1
    const tex = i === 0 ? tex0 : tex1
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      tex,
      0
    )
    const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER)
    if (st !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error("trail FBO incomplete")
    }
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null)

  const vs1 = compile(gl, gl.VERTEX_SHADER, FULLSCREEN_VS)
  const fsFade = compile(gl, gl.FRAGMENT_SHADER, TRAIL_FADE_FS)
  const fadeProgram = link(gl, vs1, fsFade)
  const vs2 = compile(gl, gl.VERTEX_SHADER, FULLSCREEN_VS)
  const fsComposite = compile(gl, gl.FRAGMENT_SHADER, TRAIL_COMPOSITE_FS)
  const compositeProgram = link(gl, vs2, fsComposite)

  const emptyVao = gl.createVertexArray()!
  gl.bindVertexArray(emptyVao)
  gl.bindVertexArray(null)

  return {
    fbos: [fbo0, fbo1],
    texs: [tex0, tex1],
    ping: 0,
    w,
    h,
    fadeProgram,
    compositeProgram,
    emptyVao,
    uFadePrev: gl.getUniformLocation(fadeProgram, "u_prev")!,
    uFadeBg: gl.getUniformLocation(fadeProgram, "u_bg")!,
    uFadeKeep: gl.getUniformLocation(fadeProgram, "u_keep")!,
    uCompositeTex: gl.getUniformLocation(compositeProgram, "u_tex")!,
    uCompositeView: gl.getUniformLocation(compositeProgram, "u_view")!,
    uCompositeWorldHalf: gl.getUniformLocation(
      compositeProgram,
      "u_world_half"
    )!,
    uCompositeCam: gl.getUniformLocation(compositeProgram, "u_cam")!,
    uCompositeTrailHalf: gl.getUniformLocation(
      compositeProgram,
      "u_trail_half"
    )!,
    uCompositeBg: gl.getUniformLocation(compositeProgram, "u_bg")!,
    uCompositeFoldTorus: gl.getUniformLocation(
      compositeProgram,
      "u_fold_torus"
    )!,
    uCompositeTorusPeriod: gl.getUniformLocation(
      compositeProgram,
      "u_torus_period"
    )!,
  }
}

function clearTrailFbOs(gl: WebGL2RenderingContext, bundle: TrailBundle) {
  gl.clearColor(BG_RGB[0], BG_RGB[1], BG_RGB[2], 1)
  for (const fbo of bundle.fbos) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
    gl.clear(gl.COLOR_BUFFER_BIT)
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null)
}

function ensureTrailBundle(pr: ParticleGL, w: number, h: number) {
  if (w <= 0 || h <= 0) return
  const gl = pr.gl
  if (pr.trail && pr.trail.w === w && pr.trail.h === h) return
  if (pr.trail) destroyTrailBundle(gl, pr.trail)
  pr.trail = createTrailBundle(gl, w, h)
  clearTrailFbOs(gl, pr.trail)
}

function hexToRgb01(hex: string): [number, number, number] {
  const s = hex.replace("#", "").trim()
  const v =
    s.length === 3
      ? parseInt(
          s[0] + s[0] + s[1] + s[1] + s[2] + s[2],
          16
        )
      : parseInt(s, 16)
  const r = ((v >> 16) & 255) / 255
  const g = ((v >> 8) & 255) / 255
  const b = (v & 255) / 255
  return [r, g, b]
}

export function createPaletteTexture(
  gl: WebGL2RenderingContext,
  colors: string[],
  K: number
): WebGLTexture {
  const tex = gl.createTexture()!
  gl.bindTexture(gl.TEXTURE_2D, tex)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

  const w = 256
  const data = new Uint8Array(w * 4)
  for (let i = 0; i < w; i++) {
    const ci = i < K ? i % colors.length : 0
    const [r, g, b] = hexToRgb01(colors[ci])
    data[i * 4] = Math.round(r * 255)
    data[i * 4 + 1] = Math.round(g * 255)
    data[i * 4 + 2] = Math.round(b * 255)
    data[i * 4 + 3] = 255
  }
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    w,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  )
  gl.bindTexture(gl.TEXTURE_2D, null)
  return tex
}

export function updatePaletteTexture(
  gl: WebGL2RenderingContext,
  tex: WebGLTexture,
  colors: string[],
  K: number
) {
  gl.bindTexture(gl.TEXTURE_2D, tex)
  const w = 256
  const data = new Uint8Array(w * 4)
  for (let i = 0; i < w; i++) {
    const ci = i < K ? i % colors.length : 0
    const [r, g, b] = hexToRgb01(colors[ci])
    data[i * 4] = Math.round(r * 255)
    data[i * 4 + 1] = Math.round(g * 255)
    data[i * 4 + 2] = Math.round(b * 255)
    data[i * 4 + 3] = 255
  }
  gl.texSubImage2D(
    gl.TEXTURE_2D,
    0,
    0,
    0,
    w,
    1,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  )
  gl.bindTexture(gl.TEXTURE_2D, null)
}

export function createParticleGL(
  canvas: HTMLCanvasElement,
  maxN: number
): ParticleGL {
  const gl = canvas.getContext("webgl2", {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
    powerPreference: "high-performance",
  })
  if (!gl) throw new Error("WebGL2 not available")

  const vs = compile(gl, gl.VERTEX_SHADER, VS)
  const fs = compile(gl, gl.FRAGMENT_SHADER, FS)
  const program = link(gl, vs, fs)

  const vsAccum = compile(gl, gl.VERTEX_SHADER, TRAIL_ACCUM_VS)
  const fsAccum = compile(gl, gl.FRAGMENT_SHADER, PARTICLE_POINT_FS)
  const trailAccumProgram = link(gl, vsAccum, fsAccum)

  const vao = gl.createVertexArray()!
  gl.bindVertexArray(vao)

  const buffer = gl.createBuffer()!
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
  gl.bufferData(gl.ARRAY_BUFFER, maxN * 3 * 4, gl.DYNAMIC_DRAW)

  const stride = 12
  const aPosition = 0
  const aType = 1
  gl.enableVertexAttribArray(aPosition)
  gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, stride, 0)
  gl.enableVertexAttribArray(aType)
  gl.vertexAttribPointer(aType, 1, gl.FLOAT, false, stride, 8)

  gl.bindVertexArray(null)

  const locs = {
    aPosition,
    aType,
    uCanvas: gl.getUniformLocation(program, "u_canvas")!,
    uView: gl.getUniformLocation(program, "u_view")!,
    uWorldHalf: gl.getUniformLocation(program, "u_world_half")!,
    uCam: gl.getUniformLocation(program, "u_cam")!,
    uTileShift: gl.getUniformLocation(program, "u_tile_shift")!,
    uPointPx: gl.getUniformLocation(program, "u_pointPx")!,
    uPalette: gl.getUniformLocation(program, "u_palette")!,
  }

  const locsAccum = {
    uTileShift: gl.getUniformLocation(trailAccumProgram, "u_tile_shift")!,
    uTrailHalf: gl.getUniformLocation(trailAccumProgram, "u_trail_half")!,
    uPointPx: gl.getUniformLocation(trailAccumProgram, "u_pointPx")!,
    uDpr: gl.getUniformLocation(trailAccumProgram, "u_dpr")!,
    uWorldHalfRef: gl.getUniformLocation(trailAccumProgram, "u_world_half_ref")!,
    uDrawBuf: gl.getUniformLocation(trailAccumProgram, "u_draw_buf")!,
    uPalette: gl.getUniformLocation(trailAccumProgram, "u_palette")!,
  }

  const paletteTex = createPaletteTexture(gl, ["#ffffff"], 1)

  gl.disable(gl.DEPTH_TEST)
  gl.enable(gl.BLEND)
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)

  return {
    gl,
    program,
    trailAccumProgram,
    vao,
    buffer,
    locs,
    locsAccum,
    maxN,
    paletteTex,
    paletteK: -1,
    trail: undefined,
    prevTileWrap: undefined,
    trailHistKey: null,
    openTrailGrid: undefined,
    trailScratch: new Float32Array(maxN * 3),
  }
}

export function ensureParticleBufferCapacity(pr: ParticleGL, n: number) {
  if (n <= pr.maxN) return
  const gl = pr.gl
  const next = Math.ceil(n * 1.125)
  gl.bindBuffer(gl.ARRAY_BUFFER, pr.buffer)
  gl.bufferData(gl.ARRAY_BUFFER, next * 3 * 4, gl.DYNAMIC_DRAW)
  gl.bindBuffer(gl.ARRAY_BUFFER, null)
  pr.maxN = next
  if (next * 3 > pr.trailScratch.length) {
    pr.trailScratch = new Float32Array(next * 3)
  }
}

/** Integer tile indices (inclusive) whose translated fundamental cells overlap the visible world rect. */
function wrappedTileIndexRange(
  panX: number,
  panY: number,
  zoom: number,
  whw: number,
  whh: number,
  margin: number
): { kxMin: number; kxMax: number; kyMin: number; kyMax: number } {
  const invz = 1 / Math.max(zoom, 1e-6)
  const visHalfW = whw * invz
  const visHalfH = whh * invz
  const x0 = panX - visHalfW - margin
  const x1 = panX + visHalfW + margin
  const y0 = panY - visHalfH - margin
  const y1 = panY + visHalfH + margin
  const Wx = 2 * whw
  const Wy = 2 * whh
  return {
    kxMin: Math.ceil((x0 - whw) / Wx),
    kxMax: Math.floor((x1 + whw) / Wx),
    kyMin: Math.ceil((y0 - whh) / Wy),
    kyMax: Math.floor((y1 + whh) / Wy),
  }
}

export function drawParticles(
  pr: ParticleGL,
  interleaved: Float32Array,
  nDraw: number,
  layout: ViewLayout,
  camera: ViewCamera,
  K: number,
  typeColors: string[],
  pointPxCss = 1.75,
  opts?: {
    trails?: boolean
    trailPersistence?: number
    tileWrap?: boolean
    /** Bump when sim resets (e.g. seed) to clear trails and open-world chunks. */
    trailHistKey?: number
  }
) {
  const { gl, program, vao, buffer, paletteTex } = pr
  if (K !== pr.paletteK) {
    updatePaletteTexture(gl, paletteTex, typeColors, K)
    pr.paletteK = K
  }

  const bw = gl.drawingBufferWidth
  const bh = gl.drawingBufferHeight

  if (
    opts?.trailHistKey !== undefined &&
    opts.trailHistKey !== pr.trailHistKey
  ) {
    pr.trailHistKey = opts.trailHistKey
    pr.openTrailGrid?.clearAll()
    if (pr.trail) clearTrailFbOs(gl, pr.trail)
  }

  const tileWrap = !!opts?.tileWrap
  if (pr.prevTileWrap !== undefined && pr.prevTileWrap !== tileWrap) {
    pr.openTrailGrid?.dispose()
    pr.openTrailGrid = undefined
    if (pr.trail) clearTrailFbOs(gl, pr.trail)
  }
  pr.prevTileWrap = tileWrap

  const whw = layout.worldHalfW
  const whh = layout.worldHalfH
  const Wx = 2 * whw
  const Wy = 2 * whh
  const marginWorld = Math.max(
    0.06,
    (pointPxCss / Math.max(layout.viewW, 120)) * (2 * whw)
  )

  const submitPoints = () => {
    gl.useProgram(program)
    gl.bindVertexArray(vao)
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, interleaved.subarray(0, nDraw * 3))
    gl.uniform3f(
      pr.locs.uCanvas,
      layout.width,
      layout.height,
      layout.dpr
    )
    gl.uniform4f(
      pr.locs.uView,
      layout.viewX,
      layout.viewY,
      layout.viewW,
      layout.viewH
    )
    gl.uniform2f(pr.locs.uWorldHalf, whw, whh)
    gl.uniform3f(pr.locs.uCam, camera.panX, camera.panY, camera.zoom)
    gl.uniform1f(pr.locs.uPointPx, pointPxCss)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, paletteTex)
    gl.uniform1i(pr.locs.uPalette, 0)

    if (tileWrap) {
      let { kxMin, kxMax, kyMin, kyMax } = wrappedTileIndexRange(
        camera.panX,
        camera.panY,
        camera.zoom,
        whw,
        whh,
        marginWorld
      )
      if (kxMax < kxMin) {
        kxMin = kxMax = 0
      }
      if (kyMax < kyMin) {
        kyMin = kyMax = 0
      }
      for (let ky = kyMin; ky <= kyMax; ky++) {
        for (let kx = kxMin; kx <= kxMax; kx++) {
          gl.uniform2f(pr.locs.uTileShift, kx * Wx, ky * Wy)
          gl.drawArrays(gl.POINTS, 0, nDraw)
        }
      }
    } else {
      gl.uniform2f(pr.locs.uTileShift, 0, 0)
      gl.drawArrays(gl.POINTS, 0, nDraw)
    }
    gl.bindVertexArray(null)
  }

  if (opts?.trails) {
    const keep = Math.max(
      0.75,
      Math.min(0.98, opts.trailPersistence ?? 0.9)
    )
    if (tileWrap) {
      if (pr.openTrailGrid) {
        pr.openTrailGrid.dispose()
        pr.openTrailGrid = undefined
      }
      const trailHalfW = whw * TRAIL_TORUS_HALF_MULT
      const trailHalfH = whh * TRAIL_TORUS_HALF_MULT
      ensureTrailBundle(pr, bw, bh)
      const tr = pr.trail!
      const read = tr.ping
      const write = (1 - tr.ping) as 0 | 1

      gl.bindFramebuffer(gl.FRAMEBUFFER, tr.fbos[write])
      gl.viewport(0, 0, bw, bh)
      gl.disable(gl.BLEND)
      gl.useProgram(tr.fadeProgram)
      gl.bindVertexArray(tr.emptyVao)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, tr.texs[read])
      gl.uniform1i(tr.uFadePrev, 0)
      gl.uniform3f(tr.uFadeBg, BG_RGB[0], BG_RGB[1], BG_RGB[2])
      gl.uniform1f(tr.uFadeKeep, keep)
      gl.drawArrays(gl.TRIANGLES, 0, 3)

      gl.enable(gl.BLEND)
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
      gl.useProgram(pr.trailAccumProgram)
      gl.bindVertexArray(vao)
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, interleaved.subarray(0, nDraw * 3))
      gl.uniform2f(pr.locsAccum.uTileShift, 0, 0)
      gl.uniform2f(pr.locsAccum.uTrailHalf, trailHalfW, trailHalfH)
      gl.uniform1f(pr.locsAccum.uPointPx, pointPxCss)
      gl.uniform1f(pr.locsAccum.uDpr, layout.dpr)
      gl.uniform1f(pr.locsAccum.uWorldHalfRef, whw)
      gl.uniform2f(pr.locsAccum.uDrawBuf, bw, bh)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, paletteTex)
      gl.uniform1i(pr.locsAccum.uPalette, 0)
      gl.drawArrays(gl.POINTS, 0, nDraw)
      gl.bindVertexArray(null)

      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      gl.viewport(0, 0, bw, bh)
      gl.disable(gl.BLEND)
      gl.useProgram(tr.compositeProgram)
      gl.bindVertexArray(tr.emptyVao)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, tr.texs[write])
      gl.uniform1i(tr.uCompositeTex, 0)
      gl.uniform4f(
        tr.uCompositeView,
        layout.viewX,
        layout.viewY,
        layout.viewW,
        layout.viewH
      )
      gl.uniform2f(tr.uCompositeWorldHalf, whw, whh)
      gl.uniform3f(tr.uCompositeCam, camera.panX, camera.panY, camera.zoom)
      gl.uniform2f(tr.uCompositeTrailHalf, trailHalfW, trailHalfH)
      gl.uniform3f(tr.uCompositeBg, BG_RGB[0], BG_RGB[1], BG_RGB[2])
      gl.uniform1i(tr.uCompositeFoldTorus, 1)
      gl.uniform2f(tr.uCompositeTorusPeriod, Wx, Wy)
      gl.drawArrays(gl.TRIANGLES, 0, 3)

      tr.ping = write
    } else {
      if (pr.trail) {
        destroyTrailBundle(gl, pr.trail)
        pr.trail = undefined
      }
      if (!pr.openTrailGrid) {
        pr.openTrailGrid = new OpenTrailChunkGrid(gl, PARTICLE_POINT_FS)
      }
      pr.openTrailGrid.render({
        interleaved,
        nDraw,
        layout,
        camera,
        whw,
        whh,
        bw,
        bh,
        pointPxCss,
        keep,
        marginWorld,
        paletteTex,
        vao,
        buffer,
        scratch: pr.trailScratch,
      })
    }
  } else {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    gl.viewport(0, 0, bw, bh)
    gl.clearColor(BG_RGB[0], BG_RGB[1], BG_RGB[2], 1)
    gl.clear(gl.COLOR_BUFFER_BIT)
    gl.enable(gl.BLEND)
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
    submitPoints()
  }
}

export function resizeDrawingSurface(
  canvas: HTMLCanvasElement,
  layout: ViewLayout
) {
  canvas.style.imageRendering = "pixelated"
  ;(canvas.style as CSSStyleDeclaration & { msInterpolationMode?: string }).msInterpolationMode = "nearest-neighbor"
  const bw = Math.floor(layout.width * layout.dpr)
  const bh = Math.floor(layout.height * layout.dpr)
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw
    canvas.height = bh
  }
}
