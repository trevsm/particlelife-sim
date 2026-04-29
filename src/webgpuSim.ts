import type { Sim } from "./simTypes"

const MAX_K = 50
const UNIFORM_BYTE_LENGTH = 256
const READBACK_RING_COUNT = 3
const FORCE_UNIT_SCALE = 2 / 1280

function maxRMx(RMx: number[][]): number {
  let m = 0.1
  for (const row of RMx) {
    for (const v of row) {
      if (Number.isFinite(v) && v > m) m = v
    }
  }
  return m
}

/** clear_grid → build_grid → clear_acc → apply_forces → integrate.
 *  One bind group; spatial grid is a GPU-built atomic linked list (no CPU upload).
 */
const SIM_SHADER = `
const MAX_K: u32 = 50u;
const FSCALE: f32 = 65536.0;

struct Uni {
  n: u32,
  k: u32,
  grid_dim_x: u32,
  grid_dim_y: u32,
  cells: u32,
  reach: u32,
  dt: f32,
  friction: f32,
  v_max: f32,
  force_factor: f32,
  repel: f32,
  settle_r: f32,
  settle_k: f32,
  flags: u32,
  _pa: u32,
  world_half_x: f32,
  world_half_y: f32,
}

@group(0) @binding(0) var<uniform> uni: Uni;
@group(0) @binding(1) var<storage, read_write> pos: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> ptype: array<u32>;
@group(0) @binding(4) var<storage, read_write> cell_head: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> link_next: array<i32>;
@group(0) @binding(6) var<storage, read> a_mat: array<f32>;
@group(0) @binding(7) var<storage, read_write> acc_x: array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> acc_y: array<atomic<i32>>;
@group(0) @binding(9) var<storage, read> rmin_mat: array<f32>;
@group(0) @binding(10) var<storage, read> rmax_mat: array<f32>;

fn clamp_f(x: f32, lo: f32, hi: f32) -> f32 { return min(hi, max(lo, x)); }

fn smoothstep01(t: f32) -> f32 {
  let x = clamp_f(t, 0.0, 1.0);
  return x * x * (3.0 - 2.0 * x);
}

fn accel_mag(a: f32, r: f32, r_min: f32, r_max: f32, repel: f32) -> f32 {
  if (r <= 0.0) { return 0.0; }
  if (r < r_min) { return (r / r_min) * repel - repel; }
  if (r > r_max) { return 0.0; }
  let denom = max(1e-6, r_max - r_min);
  return a * (1.0 - abs(r_min + r_max - 2.0 * r) / denom);
}

fn torus_delta_x(a: f32, b: f32) -> f32 {
  var d = b - a;
  let period = uni.world_half_x * 2.0;
  let half_p = period * 0.5;
  if (d > half_p) { d = d - period; }
  else if (d < -half_p) { d = d + period; }
  return d;
}

fn torus_delta_y(a: f32, b: f32) -> f32 {
  var d = b - a;
  let period = uni.world_half_y * 2.0;
  let half_p = period * 0.5;
  if (d > half_p) { d = d - period; }
  else if (d < -half_p) { d = d + period; }
  return d;
}

fn cell_xy(x: f32, y: f32, gdx: u32, gdy: u32) -> vec2<u32> {
  let wx = uni.world_half_x * 2.0;
  let wy = uni.world_half_y * 2.0;
  let gx = u32(clamp_f(floor((x + uni.world_half_x) / wx * f32(gdx)), 0.0, f32(gdx) - 1.0));
  let gy = u32(clamp_f(floor((y + uni.world_half_y) / wy * f32(gdy)), 0.0, f32(gdy) - 1.0));
  return vec2<u32>(gx, gy);
}

fn imod(n: i32, m: i32) -> i32 {
  let r = n % m;
  return select(r + m, r, r >= 0);
}

fn a_get(ti: u32, tj: u32) -> f32 {
  return a_mat[ti * MAX_K + tj];
}

fn rmin_get(ti: u32, tj: u32) -> f32 {
  return rmin_mat[ti * MAX_K + tj];
}

fn rmax_get(ti: u32, tj: u32) -> f32 {
  return rmax_mat[ti * MAX_K + tj];
}

@compute @workgroup_size(256)
fn clear_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= uni.cells) { return; }
  atomicStore(&cell_head[i], -1);
}

@compute @workgroup_size(256)
fn build_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= uni.n) { return; }
  let cc = cell_xy(pos[i].x, pos[i].y, uni.grid_dim_x, uni.grid_dim_y);
  let cidx = cc.x + cc.y * uni.grid_dim_x;
  let prev = atomicExchange(&cell_head[cidx], i32(i));
  link_next[i] = prev;
}

@compute @workgroup_size(256)
fn clear_acc(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= uni.n) { return; }
  atomicStore(&acc_x[i], 0);
  atomicStore(&acc_y[i], 0);
}

@compute @workgroup_size(256)
fn apply_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= uni.n) { return; }

  let gdx = uni.grid_dim_x;
  let gdy = uni.grid_dim_y;
  let reach = i32(uni.reach);
  let wrap = (uni.flags & 1u) != 0u;
  let mutual_only = (uni.flags & 2u) != 0u;
  let settle_on = (uni.flags & 4u) != 0u;

  let ti_u = ptype[i];
  let xi = pos[i].x;
  let yi = pos[i].y;
  let vxi = vel[i].x;
  let vyi = vel[i].y;
  let ARi = ti_u * MAX_K;

  var lfx = 0.0;
  var lfy = 0.0;

  let cc = cell_xy(xi, yi, gdx, gdy);
  let cx = i32(cc.x);
  let cy = i32(cc.y);

  for (var dy = -reach; dy <= reach; dy++) {
    for (var dx = -reach; dx <= reach; dx++) {
      var nx = cx + dx;
      var ny = cy + dy;
      if (wrap) {
        nx = imod(nx, i32(gdx));
        ny = imod(ny, i32(gdy));
      } else {
        if (nx < 0 || ny < 0 || nx >= i32(gdx) || ny >= i32(gdy)) { continue; }
      }
      let cidx = u32(nx) + u32(ny) * gdx;

      var cur = atomicLoad(&cell_head[cidx]);
      while (cur != -1) {
        let j = u32(cur);
        if (j != i) {
          var dxp = pos[j].x - xi;
          var dyp = pos[j].y - yi;
          if (wrap) {
            dxp = torus_delta_x(xi, pos[j].x);
            dyp = torus_delta_y(yi, pos[j].y);
          }

          let r2 = dxp * dxp + dyp * dyp;
          if (r2 > 0.0) {
            let rlen = sqrt(r2);

            let tj = ptype[j];
            let r_ij = rmin_get(ti_u, tj);
            let R_ij = rmax_get(ti_u, tj);
            if (rlen <= R_ij) {
            let invr = 1.0 / rlen;
            let ux = dxp * invr;
            let uy = dyp * invr;

            var am = a_mat[ARi + tj];
            if (mutual_only) {
              let mp = am > 0.0 && a_get(tj, ti_u) > 0.0;
              if (!mp) { am = 0.0; }
            }

            let f_ij = accel_mag(am, rlen, r_ij, R_ij, uni.repel);

            if (f_ij != 0.0) {
              lfx += f_ij * ux;
              lfy += f_ij * uy;
            }

            if (settle_on) {
              let mutual_pos = am > 0.0 && a_get(tj, ti_u) > 0.0;
              let r_s = min(r_ij, rmin_get(tj, ti_u));
              if (mutual_pos && rlen > r_s && rlen < uni.settle_r) {
                let vxj = vel[j].x;
                let vyj = vel[j].y;
                let vrel = (vxi - vxj) * ux + (vyi - vyj) * uy;
                let c_crit = 2.0 / uni.dt;
                let c = min(max(uni.settle_k, 0.0), c_crit);
                let tt = (rlen - r_s) / max(1e-6, uni.settle_r - r_s);
                let w = 1.0 - smoothstep01(tt);
                if (w > 0.0) {
                  let f_d = -c * vrel * w;
                  let sfx = f_d * ux;
                  let sfy = f_d * uy;
                  lfx += sfx;
                  lfy += sfy;
                }
              }
            }
            }
          }
        }
        cur = link_next[j];
      }
    }
  }

  let qix = i32(clamp_f(lfx * FSCALE, -2e8, 2e8));
  let qiy = i32(clamp_f(lfy * FSCALE, -2e8, 2e8));
  atomicStore(&acc_x[i], qix);
  atomicStore(&acc_y[i], qiy);
}

@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= uni.n) { return; }

  let wrap = (uni.flags & 1u) != 0u;
  let fxs = f32(atomicLoad(&acc_x[i])) / FSCALE;
  let fys = f32(atomicLoad(&acc_y[i])) / FSCALE;

  var vx = vel[i].x;
  var vy = vel[i].y;
  let dt = uni.dt;
  let friction_f = max(0.0, 1.0 - uni.friction);

  vx = (vx + fxs * uni.force_factor) * friction_f;
  vy = (vy + fys * uni.force_factor) * friction_f;

  let v2 = vx * vx + vy * vy;
  let vm = uni.v_max;
  if (v2 > vm * vm) {
    let s = vm / sqrt(v2);
    vx *= s;
    vy *= s;
  }

  var x = pos[i].x + dt * vx;
  var y = pos[i].y + dt * vy;

  let whx = uni.world_half_x;
  let why = uni.world_half_y;
  let fwx = whx * 2.0;
  let fwy = why * 2.0;

  if (wrap) {
    if (x < -whx) { x = x + fwx; }
    if (x > whx) { x = x - fwx; }
    if (y < -why) { y = y + fwy; }
    if (y > why) { y = y - fwy; }
  } else {
    if (x < -whx) {
      x = -whx + (-whx - x);
      vx = abs(vx);
    }
    if (x > whx) {
      x = whx - (x - whx);
      vx = -abs(vx);
    }
    if (y < -why) {
      y = -why + (-why - y);
      vy = abs(vy);
    }
    if (y > why) {
      y = why - (y - why);
      vy = -abs(vy);
    }
  }

  pos[i] = vec2<f32>(x, y);
  vel[i] = vec2<f32>(vx, vy);
}
`

function packUniform(sim: Sim, N: number, cells: number): ArrayBuffer {
  const sp = sim.spec
  const reach = Math.max(1, Math.ceil(maxRMx(sim.RMx) / sp.cellSize))
  const buf = new ArrayBuffer(UNIFORM_BYTE_LENGTH)
  const dv = new DataView(buf)
  let o = 0
  dv.setUint32(o, N, true)
  o += 4
  dv.setUint32(o, sim.K, true)
  o += 4
  dv.setUint32(o, sim.gridDimX, true)
  o += 4
  dv.setUint32(o, sim.gridDimY, true)
  o += 4
  dv.setUint32(o, cells, true)
  o += 4
  dv.setUint32(o, reach, true)
  o += 4
  dv.setFloat32(o, sp.dt, true)
  o += 4
  dv.setFloat32(o, sp.friction, true)
  o += 4
  dv.setFloat32(o, sp.vMax, true)
  o += 4
  dv.setFloat32(o, sp.forceFactor * FORCE_UNIT_SCALE, true)
  o += 4
  dv.setFloat32(o, sp.repel, true)
  o += 4
  dv.setFloat32(o, sp.settleR, true)
  o += 4
  dv.setFloat32(o, sp.settleK, true)
  o += 4
  let flags = 0
  if (sp.wrap) flags |= 1
  if (sp.mutualOnly) flags |= 2
  if (sp.settleEnabled) flags |= 4
  dv.setUint32(o, flags, true)
  o += 4
  dv.setUint32(o, 0, true)
  o += 4
  dv.setFloat32(o, sim.worldHalfW, true)
  o += 4
  dv.setFloat32(o, sim.worldHalfH, true)
  o += 4
  return buf
}

function flattenA(sim: Sim): Float32Array {
  const out = new Float32Array(MAX_K * MAX_K)
  const K = Math.min(sim.K, MAX_K)
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) {
      out[i * MAX_K + j] = sim.A[i][j]
    }
  }
  return out
}

function flattenMx(src: number[][]): Float32Array {
  const out = new Float32Array(MAX_K * MAX_K)
  const K = Math.min(src.length, MAX_K)
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) {
      out[i * MAX_K + j] = src[i][j]
    }
  }
  return out
}

function align256(n: number): number {
  return Math.ceil(n / 256) * 256
}

export class GpuSimRunner {
  private device: GPUDevice
  private queue: GPUQueue
  private clearGridPl: GPUComputePipeline
  private buildGridPl: GPUComputePipeline
  private clearPl: GPUComputePipeline
  private forcePl: GPUComputePipeline
  private intPl: GPUComputePipeline
  private uniformBuf: GPUBuffer
  private posBuf: GPUBuffer
  private velBuf: GPUBuffer
  private typeBuf: GPUBuffer
  private cellHeadBuf: GPUBuffer
  private nextBuf: GPUBuffer
  private aBuf: GPUBuffer
  private rMinBuf: GPUBuffer
  private rMaxBuf: GPUBuffer
  private accXBuf: GPUBuffer
  private accYBuf: GPUBuffer
  private readbackBufs: GPUBuffer[]
  private readbackInflight: boolean[]
  private nextReadback = 0
  private bindGroup: GPUBindGroup
  capN = 0
  capCells = 0
  private uniformStaging: ArrayBuffer
  private aStaging: Float32Array
  /** Avoid re-uploading static matrices every frame. */
  private aDirty = true
  private rMinDirty = true
  private rMaxDirty = true
  private lastK = -1
  /** Optional sub-stepping per render frame (1 = one physics step per frame). */
  subSteps = 1

  private constructor(
    device: GPUDevice,
    clearGridPl: GPUComputePipeline,
    buildGridPl: GPUComputePipeline,
    clearPl: GPUComputePipeline,
    forcePl: GPUComputePipeline,
    intPl: GPUComputePipeline,
    uniformBuf: GPUBuffer,
    posBuf: GPUBuffer,
    velBuf: GPUBuffer,
    typeBuf: GPUBuffer,
    cellHeadBuf: GPUBuffer,
    nextBuf: GPUBuffer,
    aBuf: GPUBuffer,
    rMinBuf: GPUBuffer,
    rMaxBuf: GPUBuffer,
    accXBuf: GPUBuffer,
    accYBuf: GPUBuffer,
    readbackBufs: GPUBuffer[],
    bindGroup: GPUBindGroup,
    capN: number,
    capCells: number
  ) {
    this.device = device
    this.queue = device.queue
    this.clearGridPl = clearGridPl
    this.buildGridPl = buildGridPl
    this.clearPl = clearPl
    this.forcePl = forcePl
    this.intPl = intPl
    this.uniformBuf = uniformBuf
    this.posBuf = posBuf
    this.velBuf = velBuf
    this.typeBuf = typeBuf
    this.cellHeadBuf = cellHeadBuf
    this.nextBuf = nextBuf
    this.aBuf = aBuf
    this.rMinBuf = rMinBuf
    this.rMaxBuf = rMaxBuf
    this.accXBuf = accXBuf
    this.accYBuf = accYBuf
    this.readbackBufs = readbackBufs
    this.readbackInflight = readbackBufs.map(() => false)
    this.bindGroup = bindGroup
    this.capN = capN
    this.capCells = capCells
    this.uniformStaging = new ArrayBuffer(UNIFORM_BYTE_LENGTH)
    this.aStaging = new Float32Array(MAX_K * MAX_K)
  }

  static async create(capN: number, gridCells: number): Promise<GpuSimRunner | null> {
    if (!("gpu" in navigator) || !navigator.gpu) return null
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    })
    if (!adapter) return null
    if (adapter.limits.maxStorageBuffersPerShaderStage < 10) {
      console.warn(
        "WebGPU sim unavailable: adapter supports fewer than 10 storage buffers per shader stage"
      )
      return null
    }
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBuffersPerShaderStage: 10,
      },
    })
    const module = device.createShaderModule({ code: SIM_SHADER })
    const bindLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        {
          binding: 6,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        {
          binding: 9,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 10,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
      ],
    })
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindLayout],
    })
    const clearGridPl = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: "clear_grid" },
    })
    const buildGridPl = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: "build_grid" },
    })
    const clearPl = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: "clear_acc" },
    })
    const forcePl = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: "apply_forces" },
    })
    const intPl = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: "integrate" },
    })

    const szN = align256(capN)
    const szVec = szN * 8
    const szU32 = align256(capN * 4)
    const szCells = align256(gridCells * 4)

    const uniformBuf = device.createBuffer({
      size: UNIFORM_BYTE_LENGTH,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const posBuf = device.createBuffer({
      size: szVec,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
    const velBuf = device.createBuffer({
      size: szVec,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
    const typeBuf = device.createBuffer({
      size: szU32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    const cellHeadBuf = device.createBuffer({
      size: szCells,
      usage: GPUBufferUsage.STORAGE,
    })
    const nextBuf = device.createBuffer({
      size: szU32,
      usage: GPUBufferUsage.STORAGE,
    })
    const aBuf = device.createBuffer({
      size: MAX_K * MAX_K * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    const rMinBuf = device.createBuffer({
      size: MAX_K * MAX_K * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    const rMaxBuf = device.createBuffer({
      size: MAX_K * MAX_K * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    const accXBuf = device.createBuffer({
      size: szU32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    const accYBuf = device.createBuffer({
      size: szU32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    const readbackBytes = szVec
    const readbackBufs: GPUBuffer[] = []
    for (let i = 0; i < READBACK_RING_COUNT; i++) {
      readbackBufs.push(
        device.createBuffer({
          size: readbackBytes,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
      )
    }

    const bindGroup = device.createBindGroup({
      layout: bindLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: posBuf } },
        { binding: 2, resource: { buffer: velBuf } },
        { binding: 3, resource: { buffer: typeBuf } },
        { binding: 4, resource: { buffer: cellHeadBuf } },
        { binding: 5, resource: { buffer: nextBuf } },
        { binding: 6, resource: { buffer: aBuf } },
        { binding: 7, resource: { buffer: accXBuf } },
        { binding: 8, resource: { buffer: accYBuf } },
        { binding: 9, resource: { buffer: rMinBuf } },
        { binding: 10, resource: { buffer: rMaxBuf } },
      ],
    })

    return new GpuSimRunner(
      device,
      clearGridPl,
      buildGridPl,
      clearPl,
      forcePl,
      intPl,
      uniformBuf,
      posBuf,
      velBuf,
      typeBuf,
      cellHeadBuf,
      nextBuf,
      aBuf,
      rMinBuf,
      rMaxBuf,
      accXBuf,
      accYBuf,
      readbackBufs,
      bindGroup,
      capN,
      gridCells
    )
  }

  /** Mark interaction matrices for re-upload next step (call on edit / preset). */
  markRulesDirty() {
    this.aDirty = true
    this.rMinDirty = true
    this.rMaxDirty = true
  }

  uploadParticleState(sim: Sim) {
    const N = Math.min(sim.spec.N, sim.x.length)
    const pv = new Float32Array(N * 2)
    const vv = new Float32Array(N * 2)
    const tt = new Uint32Array(N)
    for (let i = 0; i < N; i++) {
      pv[i * 2] = sim.x[i]
      pv[i * 2 + 1] = sim.y[i]
      vv[i * 2] = sim.vx[i]
      vv[i * 2 + 1] = sim.vy[i]
      tt[i] = sim.type[i] | 0
    }
    this.queue.writeBuffer(this.posBuf, 0, pv.buffer, pv.byteOffset, pv.byteLength)
    this.queue.writeBuffer(this.velBuf, 0, vv.buffer, vv.byteOffset, vv.byteLength)
    this.queue.writeBuffer(this.typeBuf, 0, tt)
  }

  dispose() {
    this.uniformBuf.destroy()
    this.posBuf.destroy()
    this.velBuf.destroy()
    this.typeBuf.destroy()
    this.cellHeadBuf.destroy()
    this.nextBuf.destroy()
    this.aBuf.destroy()
    this.rMinBuf.destroy()
    this.rMaxBuf.destroy()
    this.accXBuf.destroy()
    this.accYBuf.destroy()
    for (const b of this.readbackBufs) b.destroy()
  }

  /** One physics step. Sync; readback copies positions only (half bandwidth) into sim.x/y. */
  step(sim: Sim): void {
    const N = Math.min(
      sim.spec.N,
      sim.x.length,
      sim.vx.length,
      sim.type.length,
      this.capN
    )
    const cells = sim.gridDimX * sim.gridDimY
    if (cells > this.capCells) {
      console.warn("GpuSimRunner: grid overflow, skipping GPU step")
      return
    }

    const u = packUniform(sim, N, cells)
    new Uint8Array(this.uniformStaging).set(new Uint8Array(u))
    this.queue.writeBuffer(this.uniformBuf, 0, this.uniformStaging)

    if (sim.K !== this.lastK) {
      this.lastK = sim.K
      this.markRulesDirty()
    }
    if (this.aDirty) {
      const aFlat = flattenA(sim)
      this.aStaging.set(aFlat)
      this.queue.writeBuffer(
        this.aBuf,
        0,
        this.aStaging.buffer,
        this.aStaging.byteOffset,
        this.aStaging.byteLength
      )
      this.aDirty = false
    }
    if (this.rMinDirty) {
      this.aStaging.set(flattenMx(sim.rMinMx))
      this.queue.writeBuffer(
        this.rMinBuf,
        0,
        this.aStaging.buffer,
        this.aStaging.byteOffset,
        this.aStaging.byteLength
      )
      this.rMinDirty = false
    }
    if (this.rMaxDirty) {
      this.aStaging.set(flattenMx(sim.RMx))
      this.queue.writeBuffer(
        this.rMaxBuf,
        0,
        this.aStaging.buffer,
        this.aStaging.byteOffset,
        this.aStaging.byteLength
      )
      this.rMaxDirty = false
    }

    // Pick a free readback buffer; if none, this frame's positions don't reach
    // the CPU (we'll keep last-frame positions for drawing). This keeps the
    // event loop unstalled when the GPU is briefly slower than the renderer.
    let rbIdx = -1
    for (let i = 0; i < this.readbackBufs.length; i++) {
      const idx = (this.nextReadback + i) % this.readbackBufs.length
      if (!this.readbackInflight[idx]) {
        rbIdx = idx
        break
      }
    }

    const encoder = this.device.createCommandEncoder()
    const bg = this.bindGroup
    const subSteps = Math.max(1, this.subSteps | 0)

    for (let s = 0; s < subSteps; s++) {
      {
        const p = encoder.beginComputePass()
        p.setPipeline(this.clearGridPl)
        p.setBindGroup(0, bg)
        p.dispatchWorkgroups(Math.ceil(cells / 256))
        p.end()
      }
      {
        const p = encoder.beginComputePass()
        p.setPipeline(this.buildGridPl)
        p.setBindGroup(0, bg)
        p.dispatchWorkgroups(Math.ceil(N / 256))
        p.end()
      }
      {
        const p = encoder.beginComputePass()
        p.setPipeline(this.clearPl)
        p.setBindGroup(0, bg)
        p.dispatchWorkgroups(Math.ceil(N / 256))
        p.end()
      }
      {
        const p = encoder.beginComputePass()
        p.setPipeline(this.forcePl)
        p.setBindGroup(0, bg)
        p.dispatchWorkgroups(Math.ceil(N / 256))
        p.end()
      }
      {
        const p = encoder.beginComputePass()
        p.setPipeline(this.intPl)
        p.setBindGroup(0, bg)
        p.dispatchWorkgroups(Math.ceil(N / 256))
        p.end()
      }
    }

    if (rbIdx >= 0) {
      const rbBuf = this.readbackBufs[rbIdx]
      encoder.copyBufferToBuffer(this.posBuf, 0, rbBuf, 0, N * 8)
    }
    this.queue.submit([encoder.finish()])

    if (rbIdx >= 0) {
      this.readbackInflight[rbIdx] = true
      this.nextReadback = (rbIdx + 1) % this.readbackBufs.length
      const buf = this.readbackBufs[rbIdx]
      const N2 = N
      buf
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          try {
            const slice = buf.getMappedRange()
            const raw = new Float32Array(slice)
            const lim = Math.min(N2, sim.x.length)
            for (let i = 0; i < lim; i++) {
              sim.x[i] = raw[i * 2]
              sim.y[i] = raw[i * 2 + 1]
            }
            buf.unmap()
          } catch {
            // buffer may have been destroyed mid-flight; nothing to do
          } finally {
            this.readbackInflight[rbIdx] = false
          }
        })
        .catch(() => {
          this.readbackInflight[rbIdx] = false
        })
    }
    sim.frame += subSteps
  }
}

export async function createGpuSimRunner(
  maxN: number,
  maxGridCells: number
): Promise<GpuSimRunner | null> {
  try {
    return await GpuSimRunner.create(maxN, maxGridCells)
  } catch (e) {
    console.warn("WebGPU sim unavailable:", e)
    return null
  }
}
