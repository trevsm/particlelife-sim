/**
 * Your-World-of-Text-style infinite trail layer: fixed world chunks, each with its own
 * ping-pong trail texture. Visible tiles are composited; LRU evicts distant tiles.
 */

export type OTViewCamera = {
  panX: number
  panY: number
  zoom: number
}

export type OTViewLayout = {
  width: number
  height: number
  dpr: number
  viewX: number
  viewY: number
  viewW: number
  viewH: number
  worldHalfW: number
  worldHalfH: number
}
export const OPEN_TRAIL_CHUNK_WORLD = 3.4
export const OPEN_TRAIL_CHUNK_PX = 304
export const OPEN_TRAIL_MAX_CHUNKS = 192

/** Same bounds the chunk renderer uses for fade/blit (world units, floor-grid indices). */
export function openTrailVisibleChunkRange(
  cam: OTViewCamera,
  whw: number,
  whh: number,
  marginWorld: number,
  chunkWorld = OPEN_TRAIL_CHUNK_WORLD
): { ix0: number; ix1: number; iy0: number; iy1: number } {
  const C = chunkWorld
  const invz = 1 / Math.max(cam.zoom, 1e-6)
  const x0 = cam.panX - whw * invz - marginWorld
  const x1 = cam.panX + whw * invz + marginWorld
  const y0 = cam.panY - whh * invz - marginWorld
  const y1 = cam.panY + whh * invz + marginWorld
  return {
    ix0: Math.floor(x0 / C),
    ix1: Math.floor(x1 / C),
    iy0: Math.floor(y0 / C),
    iy1: Math.floor(y1 / C),
  }
}

const BG_RGB: [number, number, number] = [0.04, 0.04, 0.04]

type ChunkRec = {
  ix: number
  iy: number
  fbos: [WebGLFramebuffer, WebGLFramebuffer]
  texs: [WebGLTexture, WebGLTexture]
  ping: 0 | 1
  lruStamp: number
}

const CHUNK_ACCUM_VS = `#version 300 es
layout(location = 0) in vec2 a_position;
layout(location = 1) in float a_type;
out float v_type;
uniform vec2 u_tile_shift;
uniform vec2 u_chunk_origin;
uniform float u_chunk_world;
uniform float u_pointPx;
uniform float u_dpr;
uniform float u_world_half_ref;
uniform vec2 u_draw_buf;

void main() {
  v_type = a_type;
  vec2 wpos = a_position + u_tile_shift;
  float lx = wpos.x - u_chunk_origin.x;
  float ly = wpos.y - u_chunk_origin.y;
  float nx = lx / u_chunk_world * 2.0 - 1.0;
  float ny = - (ly / u_chunk_world * 2.0 - 1.0);
  float pxf = floor((nx * 0.5 + 0.5) * u_draw_buf.x) + 0.5;
  float pyf = floor((ny * 0.5 + 0.5) * u_draw_buf.y) + 0.5;
  float snx = (pxf / u_draw_buf.x) * 2.0 - 1.0;
  float sny = (pyf / u_draw_buf.y) * 2.0 - 1.0;
  gl_Position = vec4(snx, sny, 0.0, 1.0);
  float halfCw = u_chunk_world * 0.5;
  float scale = max(u_world_half_ref / max(halfCw, 1e-6), 0.05);
  float ps = u_pointPx * u_dpr * scale;
  gl_PointSize = max(1.0, floor(ps + 0.5));
}
`

const CHUNK_BLIT_VS = `#version 300 es
uniform vec4 u_view;
uniform vec3 u_canvas;
uniform vec2 u_world_half;
uniform vec3 u_cam;
uniform vec4 u_world_rect;
out vec2 v_uv;

vec2 worldToSnappedNdc(vec2 w) {
  float invz = 1.0 / max(u_cam.z, 1e-6);
  float hw = u_world_half.x * invz;
  float hh = u_world_half.y * invz;
  float tu = ((w.x - u_cam.x) + hw) / (2.0 * hw);
  float tv = ((w.y - u_cam.y) + hh) / (2.0 * hh);
  vec2 screen = vec2(u_view.x + tu * u_view.z, u_view.y + tv * u_view.w);
  float bufW = u_canvas.x * u_canvas.z;
  float bufH = u_canvas.y * u_canvas.z;
  float pxf = floor((screen.x / u_canvas.x) * bufW) + 0.5;
  float pyf = floor((screen.y / u_canvas.y) * bufH) + 0.5;
  return vec2((pxf / bufW) * 2.0 - 1.0, -((pyf / bufH) * 2.0 - 1.0));
}

void main() {
  float x0 = u_world_rect.x;
  float y0 = u_world_rect.y;
  float x1 = u_world_rect.z;
  float y1 = u_world_rect.w;
  vec2 w =
    gl_VertexID == 0 ? vec2(x0, y0)
    : gl_VertexID == 1 ? vec2(x1, y0)
    : gl_VertexID == 2 ? vec2(x0, y1)
    : vec2(x1, y1);
  gl_Position = vec4(worldToSnappedNdc(w), 0.0, 1.0);
  v_uv =
    gl_VertexID == 0 ? vec2(0.0, 0.0)
    : gl_VertexID == 1 ? vec2(1.0, 0.0)
    : gl_VertexID == 2 ? vec2(0.0, 1.0)
    : vec2(1.0, 1.0);
}
`

const CHUNK_BLIT_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec3 u_bg;
out vec4 outColor;

void main() {
  ivec2 ts = textureSize(u_tex, 0);
  int tx = clamp(int(floor(v_uv.x * float(ts.x))), 0, ts.x - 1);
  // Accum VS maps small chunk ly to top of FBO (high ty in texelFetch coords);
  // v_uv.y=0 is world y0 → must sample high ty. Flip V to match FBO layout.
  float vf = 1.0 - v_uv.y;
  int ty = clamp(int(floor(vf * float(ts.y))), 0, ts.y - 1);
  vec3 c = texelFetch(u_tex, ivec2(tx, ty), 0).rgb;
  float isBg = step(length(c - u_bg), 0.015);
  outColor = vec4(c, 1.0 - isBg);
}
`

const FADE_FS = `#version 300 es
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

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
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

function link(gl: WebGL2RenderingContext, vs: WebGLShader, fs: WebGLShader): WebGLProgram {
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

function chunkKey(ix: number, iy: number): string {
  return `${ix},${iy}`
}

export class OpenTrailChunkGrid {
  readonly gl: WebGL2RenderingContext
  readonly C: number
  readonly px: number
  readonly maxChunks: number
  private readonly map = new Map<string, ChunkRec>()
  private readonly emptyVao: WebGLVertexArrayObject
  private readonly fadeProg: WebGLProgram
  private readonly accumProg: WebGLProgram
  private readonly blitProg: WebGLProgram
  private uFadePrev: WebGLUniformLocation
  private uFadeBg: WebGLUniformLocation
  private uFadeKeep: WebGLUniformLocation
  private uAccumTileShift: WebGLUniformLocation
  private uAccumChunkOrigin: WebGLUniformLocation
  private uAccumChunkWorld: WebGLUniformLocation
  private uAccumPointPx: WebGLUniformLocation
  private uAccumDpr: WebGLUniformLocation
  private uAccumWhRef: WebGLUniformLocation
  private uAccumDrawBuf: WebGLUniformLocation
  private uAccumPalette: WebGLUniformLocation
  private uBlitView: WebGLUniformLocation
  private uBlitCanvas: WebGLUniformLocation
  private uBlitWh: WebGLUniformLocation
  private uBlitCam: WebGLUniformLocation
  private uBlitWorldRect: WebGLUniformLocation
  private uBlitTex: WebGLUniformLocation
  private uBlitBg: WebGLUniformLocation
  private lruSeq = 0

  constructor(
    gl: WebGL2RenderingContext,
    particleFragmentSrc: string,
    chunkWorld = OPEN_TRAIL_CHUNK_WORLD,
    chunkPx = OPEN_TRAIL_CHUNK_PX,
    maxChunks = OPEN_TRAIL_MAX_CHUNKS
  ) {
    this.gl = gl
    this.C = chunkWorld
    this.px = chunkPx
    this.maxChunks = maxChunks

    const vsAccum = compile(gl, gl.VERTEX_SHADER, CHUNK_ACCUM_VS)
    const fsPart = compile(gl, gl.FRAGMENT_SHADER, particleFragmentSrc)
    this.accumProg = link(gl, vsAccum, fsPart)

    const vsFade = compile(gl, gl.VERTEX_SHADER, FULLSCREEN_VS)
    const fsFade = compile(gl, gl.FRAGMENT_SHADER, FADE_FS)
    this.fadeProg = link(gl, vsFade, fsFade)

    const vsBlit = compile(gl, gl.VERTEX_SHADER, CHUNK_BLIT_VS)
    const fsBlit = compile(gl, gl.FRAGMENT_SHADER, CHUNK_BLIT_FS)
    this.blitProg = link(gl, vsBlit, fsBlit)

    this.emptyVao = gl.createVertexArray()!
    gl.bindVertexArray(this.emptyVao)
    gl.bindVertexArray(null)

    this.uFadePrev = gl.getUniformLocation(this.fadeProg, "u_prev")!
    this.uFadeBg = gl.getUniformLocation(this.fadeProg, "u_bg")!
    this.uFadeKeep = gl.getUniformLocation(this.fadeProg, "u_keep")!

    this.uAccumTileShift = gl.getUniformLocation(this.accumProg, "u_tile_shift")!
    this.uAccumChunkOrigin = gl.getUniformLocation(this.accumProg, "u_chunk_origin")!
    this.uAccumChunkWorld = gl.getUniformLocation(this.accumProg, "u_chunk_world")!
    this.uAccumPointPx = gl.getUniformLocation(this.accumProg, "u_pointPx")!
    this.uAccumDpr = gl.getUniformLocation(this.accumProg, "u_dpr")!
    this.uAccumWhRef = gl.getUniformLocation(this.accumProg, "u_world_half_ref")!
    this.uAccumDrawBuf = gl.getUniformLocation(this.accumProg, "u_draw_buf")!
    this.uAccumPalette = gl.getUniformLocation(this.accumProg, "u_palette")!

    this.uBlitView = gl.getUniformLocation(this.blitProg, "u_view")!
    this.uBlitCanvas = gl.getUniformLocation(this.blitProg, "u_canvas")!
    this.uBlitWh = gl.getUniformLocation(this.blitProg, "u_world_half")!
    this.uBlitCam = gl.getUniformLocation(this.blitProg, "u_cam")!
    this.uBlitWorldRect = gl.getUniformLocation(this.blitProg, "u_world_rect")!
    this.uBlitTex = gl.getUniformLocation(this.blitProg, "u_tex")!
    this.uBlitBg = gl.getUniformLocation(this.blitProg, "u_bg")!
  }

  dispose() {
    const gl = this.gl
    for (const ch of this.map.values()) {
      gl.deleteFramebuffer(ch.fbos[0])
      gl.deleteFramebuffer(ch.fbos[1])
      gl.deleteTexture(ch.texs[0])
      gl.deleteTexture(ch.texs[1])
    }
    this.map.clear()
    gl.deleteProgram(this.fadeProg)
    gl.deleteProgram(this.accumProg)
    gl.deleteProgram(this.blitProg)
    gl.deleteVertexArray(this.emptyVao)
  }

  clearAll() {
    const gl = this.gl
    for (const ch of this.map.values()) {
      gl.deleteFramebuffer(ch.fbos[0])
      gl.deleteFramebuffer(ch.fbos[1])
      gl.deleteTexture(ch.texs[0])
      gl.deleteTexture(ch.texs[1])
    }
    this.map.clear()
  }

  private makeChunkTextures(): [WebGLTexture, WebGLTexture] {
    const gl = this.gl
    const px = this.px
    const makeTex = (): WebGLTexture => {
      const tex = gl.createTexture()!
      gl.bindTexture(gl.TEXTURE_2D, tex)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, px, px, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
      gl.bindTexture(gl.TEXTURE_2D, null)
      return tex
    }
    return [makeTex(), makeTex()]
  }

  private evictOne() {
    const gl = this.gl
    let bestK = ""
    let bestS = Infinity
    for (const [k, ch] of this.map) {
      if (ch.lruStamp < bestS) {
        bestS = ch.lruStamp
        bestK = k
      }
    }
    if (!bestK) return
    const ch = this.map.get(bestK)!
    gl.deleteFramebuffer(ch.fbos[0])
    gl.deleteFramebuffer(ch.fbos[1])
    gl.deleteTexture(ch.texs[0])
    gl.deleteTexture(ch.texs[1])
    this.map.delete(bestK)
  }

  private ensureChunk(ix: number, iy: number): ChunkRec {
    const k = chunkKey(ix, iy)
    let ch = this.map.get(k)
    if (ch) {
      ch.lruStamp = ++this.lruSeq
      return ch
    }
    while (this.map.size >= this.maxChunks) {
      this.evictOne()
    }
    const gl = this.gl
    const [t0, t1] = this.makeChunkTextures()
    const f0 = gl.createFramebuffer()!
    const f1 = gl.createFramebuffer()!
    for (let i = 0; i < 2; i++) {
      const f = i === 0 ? f0 : f1
      const t = i === 0 ? t0 : t1
      gl.bindFramebuffer(gl.FRAMEBUFFER, f)
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        t,
        0
      )
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error("open trail chunk FBO incomplete")
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    ch = {
      ix,
      iy,
      fbos: [f0, f1],
      texs: [t0, t1],
      ping: 0,
      lruStamp: ++this.lruSeq,
    }
    this.clearChunkPair(ch)
    this.map.set(k, ch)
    return ch
  }

  private clearChunkPair(ch: ChunkRec) {
    const gl = this.gl
    gl.clearColor(BG_RGB[0], BG_RGB[1], BG_RGB[2], 1)
    for (const fbo of ch.fbos) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
      gl.clear(gl.COLOR_BUFFER_BIT)
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
  }

  /** Chunk index for world coordinate (lower-left grid corner at ix*C, iy*C). */
  static cellIndex(v: number, C: number): number {
    return Math.floor(v / C)
  }

  visibleChunkRange(
    cam: OTViewCamera,
    whw: number,
    whh: number,
    margin: number
  ): { ix0: number; ix1: number; iy0: number; iy1: number } {
    return openTrailVisibleChunkRange(cam, whw, whh, margin, this.C)
  }

  render(params: {
    interleaved: Float32Array
    nDraw: number
    layout: OTViewLayout
    camera: OTViewCamera
    whw: number
    whh: number
    bw: number
    bh: number
    pointPxCss: number
    keep: number
    marginWorld: number
    paletteTex: WebGLTexture
    /** Main particle VAO + buffer + count for splatting subranges */
    vao: WebGLVertexArrayObject
    buffer: WebGLBuffer
    scratch: Float32Array
  }) {
    const gl = this.gl
    const {
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
      scratch,
    } = params

    const C = this.C
    const px = this.px
    const { ix0, ix1, iy0, iy1 } = this.visibleChunkRange(
      camera,
      whw,
      whh,
      marginWorld
    )

    const bucket = new Map<string, number[]>()
    for (let i = 0; i < nDraw; i++) {
      const x = interleaved[i * 3]
      const y = interleaved[i * 3 + 1]
      const ix = OpenTrailChunkGrid.cellIndex(x, C)
      const iy = OpenTrailChunkGrid.cellIndex(y, C)
      const k = chunkKey(ix, iy)
      let arr = bucket.get(k)
      if (!arr) {
        arr = []
        bucket.set(k, arr)
      }
      arr.push(i)
    }

    const toProcess = new Set<string>(bucket.keys())
    for (let ix = ix0; ix <= ix1; ix++) {
      for (let iy = iy0; iy <= iy1; iy++) {
        toProcess.add(chunkKey(ix, iy))
      }
    }

    for (const k of toProcess) {
      const [sx, sy] = k.split(",").map(Number)
      const ch = this.ensureChunk(sx, sy)
      const read = ch.ping
      const write = (1 - ch.ping) as 0 | 1

      gl.bindFramebuffer(gl.FRAMEBUFFER, ch.fbos[write])
      gl.viewport(0, 0, px, px)
      gl.disable(gl.BLEND)
      gl.useProgram(this.fadeProg)
      gl.bindVertexArray(this.emptyVao)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, ch.texs[read])
      gl.uniform1i(this.uFadePrev, 0)
      gl.uniform3f(this.uFadeBg, BG_RGB[0], BG_RGB[1], BG_RGB[2])
      gl.uniform1f(this.uFadeKeep, keep)
      gl.drawArrays(gl.TRIANGLES, 0, 3)

      const idxs = bucket.get(k)
      if (idxs && idxs.length > 0) {
        let w = 0
        for (const pi of idxs) {
          const o = w * 3
          scratch[o] = interleaved[pi * 3]
          scratch[o + 1] = interleaved[pi * 3 + 1]
          scratch[o + 2] = interleaved[pi * 3 + 2]
          w++
        }
        gl.enable(gl.BLEND)
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
        gl.useProgram(this.accumProg)
        gl.bindVertexArray(vao)
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, scratch.subarray(0, w * 3))
        gl.uniform2f(this.uAccumTileShift, 0, 0)
        gl.uniform2f(this.uAccumChunkOrigin, sx * C, sy * C)
        gl.uniform1f(this.uAccumChunkWorld, C)
        gl.uniform1f(this.uAccumPointPx, pointPxCss)
        gl.uniform1f(this.uAccumDpr, layout.dpr)
        gl.uniform1f(this.uAccumWhRef, whw)
        gl.uniform2f(this.uAccumDrawBuf, px, px)
        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, paletteTex)
        gl.uniform1i(this.uAccumPalette, 0)
        gl.drawArrays(gl.POINTS, 0, w)
      }

      ch.ping = write
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    gl.viewport(0, 0, bw, bh)
    gl.clearColor(BG_RGB[0], BG_RGB[1], BG_RGB[2], 1)
    gl.clear(gl.COLOR_BUFFER_BIT)
    gl.enable(gl.BLEND)
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
    gl.useProgram(this.blitProg)
    gl.bindVertexArray(this.emptyVao)

    for (let ix = ix0; ix <= ix1; ix++) {
      for (let iy = iy0; iy <= iy1; iy++) {
        const ch = this.map.get(chunkKey(ix, iy))
        if (!ch) continue
        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, ch.texs[ch.ping])
        gl.uniform1i(this.uBlitTex, 0)
        gl.uniform4f(
          this.uBlitView,
          layout.viewX,
          layout.viewY,
          layout.viewW,
          layout.viewH
        )
        gl.uniform3f(this.uBlitCanvas, layout.width, layout.height, layout.dpr)
        gl.uniform2f(this.uBlitWh, whw, whh)
        gl.uniform3f(this.uBlitCam, camera.panX, camera.panY, camera.zoom)
        gl.uniform4f(
          this.uBlitWorldRect,
          ix * C,
          iy * C,
          (ix + 1) * C,
          (iy + 1) * C
        )
        gl.uniform3f(this.uBlitBg, BG_RGB[0], BG_RGB[1], BG_RGB[2])
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
      }
    }

    gl.bindVertexArray(null)
  }
}
