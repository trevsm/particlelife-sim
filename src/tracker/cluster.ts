/**
 * Cluster detection — connected components on the proximity graph.
 *
 * Reuses the sim's existing uniform grid (cellHead / next) which step(sim)
 * rebuilds each frame. Edge between particles i and j exists iff their distance
 * is below `bondFactor × rMinMx[type[i]][type[j]]` — i.e. they are within the
 * bonded-pair radius for their respective types.
 *
 * Wrap and open-world are both supported by mirroring the existing forEachNeighbor
 * walk. Costs O(N + edges) per call; integrated into the existing per-frame work.
 */

import type { Sim } from "../simTypes"
import type { Cluster, TrackerParams } from "./types"

class UnionFind {
  parent: Int32Array
  rank: Uint8Array
  constructor(n: number) {
    this.parent = new Int32Array(n)
    this.rank = new Uint8Array(n)
    for (let i = 0; i < n; i++) this.parent[i] = i
  }
  find(x: number): number {
    let r = x
    while (this.parent[r] !== r) r = this.parent[r]
    // path compression
    while (this.parent[x] !== r) {
      const next = this.parent[x]
      this.parent[x] = r
      x = next
    }
    return r
  }
  union(a: number, b: number) {
    const ra = this.find(a)
    const rb = this.find(b)
    if (ra === rb) return
    if (this.rank[ra] < this.rank[rb]) this.parent[ra] = rb
    else if (this.rank[ra] > this.rank[rb]) this.parent[rb] = ra
    else {
      this.parent[rb] = ra
      this.rank[ra]++
    }
  }
}

function torusDelta(a: number, b: number, full: number): number {
  let d = b - a
  const half = full * 0.5
  if (d > half) d -= full
  else if (d < -half) d += full
  return d
}

export function detectClusters(sim: Sim, params: TrackerParams): Cluster[] {
  const N = Math.min(sim.spec.N, sim.x.length, sim.type.length)
  const K = sim.K
  if (N === 0) return []

  const X = sim.x
  const Y = sim.y
  const TYPE = sim.type
  const RMINMX = sim.rMinMx
  const bond = params.bondFactor
  const wrap = sim.spec.wrap
  const worldW = 2 * sim.worldHalfW
  const worldH = 2 * sim.worldHalfH
  const whw = sim.worldHalfW
  const whh = sim.worldHalfH

  const cellHead = sim.cellHead
  const linkNext = sim.next
  const gdx = sim.gridDimX
  const gdy = sim.gridDimY
  const gOx = sim.gridOriginX
  const gOy = sim.gridOriginY
  const gCs = sim.gridCellEff
  const cellSize = sim.spec.cellSize

  let maxBond = 0
  for (let i = 0; i < K; i++) {
    const row = RMINMX[i]
    for (let j = 0; j < K; j++) {
      const v = row[j] * bond
      if (v > maxBond) maxBond = v
    }
  }
  const cs = wrap ? cellSize : gCs
  const reach = Math.max(1, Math.ceil(maxBond / cs))
  const maxBond2 = maxBond * maxBond

  const uf = new UnionFind(N)

  for (let i = 0; i < N; i++) {
    const xi = X[i]
    const yi = Y[i]
    const ti = TYPE[i] | 0
    const RminRi = RMINMX[ti]

    let cx: number
    let cy: number
    if (wrap) {
      const cxi = ((xi + whw) / worldW) * gdx
      const cyi = ((yi + whh) / worldH) * gdy
      cx = cxi < 0 ? 0 : cxi >= gdx ? gdx - 1 : cxi | 0
      cy = cyi < 0 ? 0 : cyi >= gdy ? gdy - 1 : cyi | 0
    } else {
      const cxi = (xi - gOx) / gCs
      const cyi = (yi - gOy) / gCs
      cx = cxi < 0 ? 0 : cxi >= gdx ? gdx - 1 : cxi | 0
      cy = cyi < 0 ? 0 : cyi >= gdy ? gdy - 1 : cyi | 0
    }

    for (let dy = -reach; dy <= reach; dy++) {
      for (let dx = -reach; dx <= reach; dx++) {
        let nx = cx + dx
        let ny = cy + dy
        if (wrap) {
          nx = ((nx % gdx) + gdx) % gdx
          ny = ((ny % gdy) + gdy) % gdy
        } else {
          if (nx < 0 || nx >= gdx || ny < 0 || ny >= gdy) continue
        }
        let j = cellHead[nx + gdx * ny]
        while (j !== -1) {
          if (j > i) {
            const ddx = wrap ? torusDelta(xi, X[j], worldW) : X[j] - xi
            const ddy = wrap ? torusDelta(yi, Y[j], worldH) : Y[j] - yi
            const r2 = ddx * ddx + ddy * ddy
            if (r2 < maxBond2) {
              const tj = TYPE[j] | 0
              const thresh = RminRi[tj] * bond
              if (r2 < thresh * thresh) {
                uf.union(i, j)
              }
            }
          }
          j = linkNext[j]
        }
      }
    }
  }

  const groups = new Map<number, number[]>()
  for (let i = 0; i < N; i++) {
    const r = uf.find(i)
    let arr = groups.get(r)
    if (!arr) {
      arr = []
      groups.set(r, arr)
    }
    arr.push(i)
  }

  const out: Cluster[] = []
  for (const members of groups.values()) {
    if (members.length < params.nMin) continue
    out.push(buildCluster(members, sim, K))
  }
  return out
}

/**
 * Build a Cluster from a member set. Handles wrap-around CoM via "unwrap relative
 * to anchor": pick the first member as reference, shift others into the same
 * spatial copy, average, then wrap CoM back into world.
 */
function buildCluster(members: number[], sim: Sim, K: number): Cluster {
  const X = sim.x
  const Y = sim.y
  const VX = sim.vx
  const VY = sim.vy
  const TYPE = sim.type
  const wrap = sim.spec.wrap
  const whw = sim.worldHalfW
  const whh = sim.worldHalfH
  const worldW = 2 * whw
  const worldH = 2 * whh
  const N = members.length

  const x0 = X[members[0]]
  const y0 = Y[members[0]]

  let sumX = 0
  let sumY = 0
  let sumVX = 0
  let sumVY = 0
  const hist = new Array<number>(K).fill(0)

  // Cache unwrapped positions for second pass (rg)
  const ux = new Float32Array(N)
  const uy = new Float32Array(N)

  for (let k = 0; k < N; k++) {
    const idx = members[k]
    let xi = X[idx]
    let yi = Y[idx]
    if (wrap) {
      const dxA = xi - x0
      if (dxA > whw) xi -= worldW
      else if (dxA < -whw) xi += worldW
      const dyA = yi - y0
      if (dyA > whh) yi -= worldH
      else if (dyA < -whh) yi += worldH
    }
    ux[k] = xi
    uy[k] = yi
    sumX += xi
    sumY += yi
    sumVX += VX[idx]
    sumVY += VY[idx]
    hist[TYPE[idx] | 0]++
  }

  const cmX = sumX / N
  const cmY = sumY / N
  const vCx = sumVX / N
  const vCy = sumVY / N

  let sumR2 = 0
  for (let k = 0; k < N; k++) {
    const dx = ux[k] - cmX
    const dy = uy[k] - cmY
    sumR2 += dx * dx + dy * dy
  }
  const rg = Math.sqrt(sumR2 / N)

  for (let i = 0; i < K; i++) hist[i] /= N

  let comX = cmX
  let comY = cmY
  if (wrap) {
    if (comX > whw) comX -= worldW
    else if (comX < -whw) comX += worldW
    if (comY > whh) comY -= worldH
    else if (comY < -whh) comY += worldH
  }

  return {
    members: new Set(members),
    com: [comX, comY],
    vCom: [vCx, vCy],
    size: N,
    rg,
    typeHistogram: hist,
  }
}
