/** WebGL2 instanced-style point sprites for many particles (1M+ draw calls → one drawArrays). */

export type ViewLayout = {
  width: number
  height: number
  dpr: number
  viewX: number
  viewY: number
  viewSize: number
  scale: number
}

export type ParticleGL = {
  gl: WebGL2RenderingContext
  program: WebGLProgram
  vao: WebGLVertexArrayObject
  buffer: WebGLBuffer
  locs: {
    aPosition: number
    aType: number
    uCanvas: WebGLUniformLocation
    uView: WebGLUniformLocation
    uPointPx: WebGLUniformLocation
    uPalette: WebGLUniformLocation
  }
  maxN: number
  paletteTex: WebGLTexture
  paletteK: number
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
uniform vec3 u_view;   // viewX, viewY, viewSize (css px)
uniform float u_pointPx;

void main() {
  v_type = a_type;
  vec2 screen = vec2(
    u_view.x + (a_position.x + 1.0) * 0.5 * u_view.z,
    u_view.y + (a_position.y + 1.0) * 0.5 * u_view.z
  );
  vec2 ndc = vec2(
    (screen.x / u_canvas.x) * 2.0 - 1.0,
    -((screen.y / u_canvas.y) * 2.0 - 1.0)
  );
  gl_Position = vec4(ndc, 0.0, 1.0);
  gl_PointSize = max(1.0, u_pointPx * u_canvas.z);
}
`

const FS = `#version 300 es
precision highp float;
in float v_type;
uniform highp sampler2D u_palette;
out vec4 outColor;

void main() {
  vec2 c = gl_PointCoord - vec2(0.5);
  if (dot(c, c) > 0.25) discard;
  int k = clamp(int(v_type + 0.5), 0, 255);
  vec3 col = texelFetch(u_palette, ivec2(k, 0), 0).rgb;
  outColor = vec4(col, 1.0);
}
`

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
    uPointPx: gl.getUniformLocation(program, "u_pointPx")!,
    uPalette: gl.getUniformLocation(program, "u_palette")!,
  }

  const paletteTex = createPaletteTexture(gl, ["#ffffff"], 1)

  gl.disable(gl.DEPTH_TEST)
  gl.enable(gl.BLEND)
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)

  return {
    gl,
    program,
    vao,
    buffer,
    locs,
    maxN,
    paletteTex,
    paletteK: -1,
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
}

export function drawParticles(
  pr: ParticleGL,
  interleaved: Float32Array,
  nDraw: number,
  layout: ViewLayout,
  K: number,
  typeColors: string[],
  pointPxCss = 1.75
) {
  const { gl, program, vao, buffer, paletteTex } = pr
  if (K !== pr.paletteK) {
    updatePaletteTexture(gl, paletteTex, typeColors, K)
    pr.paletteK = K
  }

  gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight)
  gl.clearColor(0.04, 0.04, 0.04, 1)
  gl.clear(gl.COLOR_BUFFER_BIT)

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
  gl.uniform3f(pr.locs.uView, layout.viewX, layout.viewY, layout.viewSize)
  gl.uniform1f(pr.locs.uPointPx, pointPxCss)

  gl.activeTexture(gl.TEXTURE0)
  gl.bindTexture(gl.TEXTURE_2D, paletteTex)
  gl.uniform1i(pr.locs.uPalette, 0)

  gl.drawArrays(gl.POINTS, 0, nDraw)
  gl.bindVertexArray(null)
}

export function resizeDrawingSurface(
  canvas: HTMLCanvasElement,
  layout: ViewLayout
) {
  const bw = Math.floor(layout.width * layout.dpr)
  const bh = Math.floor(layout.height * layout.dpr)
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw
    canvas.height = bh
  }
}
