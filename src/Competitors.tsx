/**
 * Competitors HUD — right-side panel listing alive organisms (top N by score)
 * and a recently-dead feed. Hover a row to highlight on the main canvas.
 *
 * Leaderboard rows use Motion: layout for rank changes, fade for enter/exit only.
 */

import { AnimatePresence, LayoutGroup, motion } from "motion/react"
import { useEffect, useRef, type CSSProperties } from "react"
import type { Organism, TrackerSnapshot } from "./tracker"

type Props = {
  snapshot: TrackerSnapshot
  stabilityGate: number
  symmetryGate: number
  leaderboardGraceSeconds: number
  topN: number
  onHover: (id: number | null) => void
  collapsed: boolean
  onToggleCollapse: () => void
}

const ACCENT = "#5DCAA5"
const FG = "#f0f0f0"
const FG2 = "rgba(255,255,255,0.6)"
const FG3 = "rgba(255,255,255,0.42)"
const FG4 = "rgba(255,255,255,0.28)"
const LINE = "rgba(255,255,255,0.08)"
const LINE2 = "rgba(255,255,255,0.16)"
const FONT_MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"

const ROW_FADE = { duration: 0.22, ease: [0.33, 1, 0.68, 1] as const }
const ROW_LAYOUT = { duration: 0.45, ease: [0.2, 0.78, 0.22, 1] as const }

/** Longer CSS edge (px) for every competitor thumbnail — bitmap is scaled with `pixelated`. */
const THUMB_CSS_LONG_EDGE = 96

/** Same box for all rows: preserve main-canvas aspect (`viewW/viewH`). */
function thumbnailCssDimensions(viewAspect: number): {
  width: number
  height: number
} {
  const a =
    viewAspect > 0 && Number.isFinite(viewAspect) ? viewAspect : 1
  if (a >= 1) {
    return { width: THUMB_CSS_LONG_EDGE, height: THUMB_CSS_LONG_EDGE / a }
  }
  return { width: THUMB_CSS_LONG_EDGE * a, height: THUMB_CSS_LONG_EDGE }
}

function AnimatedLeaderboard({
  visible,
  snapshot,
  incubating,
  stabilityGate,
  symmetryGate,
  leaderboardGraceSeconds,
  onHover,
}: {
  visible: Organism[]
  incubating: number
  stabilityGate: number
  symmetryGate: number
  leaderboardGraceSeconds: number
  snapshot: TrackerSnapshot
  onHover: (id: number | null) => void
}) {
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const thumbCss = thumbnailCssDimensions(
    snapshot.thumbnailViewAspect ?? 1
  )

  return (
    <div
      ref={scrollRef}
      style={{
        flex: 1,
        overflowY: "hidden",
        padding: "8px 8px 12px",
      }}
    >
      {visible.length === 0 && incubating === 0 && (
        <div
          style={{
            padding: 16,
            fontSize: 12,
            color: FG3,
            fontFamily: FONT_MONO,
            lineHeight: 1.5,
          }}
        >
          no organisms yet — waiting for clusters to form
        </div>
      )}

      <LayoutGroup id="competitor-leaderboard">
        <div>
          <AnimatePresence mode="popLayout" initial={false}>
            {visible.map((org, i) => (
              <motion.div
                key={org.id}
                layout
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{ marginBottom: 6 }}
                transition={{
                  opacity: ROW_FADE,
                  layout: ROW_LAYOUT,
                }}
              >
                <OrganismRow
                  org={org}
                  rank={i + 1}
                  onHover={onHover}
                  isHighlighted={snapshot.highlightId === org.id}
                  thumbCssWidth={thumbCss.width}
                  thumbCssHeight={thumbCss.height}
                />
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </LayoutGroup>

      {incubating > 0 && (
        <div
          style={{
            padding: "10px 8px 4px",
            fontSize: 11,
            fontFamily: FONT_MONO,
            color: FG3,
            letterSpacing: "0.04em",
          }}
        >
          {incubating} incubating · rolling avg stab ≥
          {(stabilityGate * 100).toFixed(0)}% · sym ≥
          {(symmetryGate * 100).toFixed(0)}% · grace{" "}
          {leaderboardGraceSeconds.toFixed(1)}s
        </div>
      )}

      {snapshot.dead.length > 0 && (
        <>
          <div
            style={{
              margin: "16px 8px 8px",
              paddingTop: 14,
              borderTop: `0.5px solid ${LINE}`,
              fontFamily: FONT_MONO,
              fontSize: 11,
              letterSpacing: "0.14em",
              color: FG3,
            }}
          >
            recently dead
          </div>
          {snapshot.dead.map((org) => (
            <DeadRow key={org.id} org={org} />
          ))}
        </>
      )}
    </div>
  )
}

export function Competitors({
  snapshot,
  stabilityGate,
  symmetryGate,
  leaderboardGraceSeconds,
  topN,
  onHover,
  collapsed,
  onToggleCollapse,
}: Props) {
  const visible = snapshot.alive
    .filter((o) => o.leaderboardListed)
    .slice(0, topN)
  const incubating = snapshot.alive.length - visible.length

  return (
    <aside
      style={{
        position: "absolute",
        top: 12,
        right: 12,
        bottom: 12,
        width: collapsed ? 36 : 320,
        background: "rgba(8,8,8,0.78)",
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
        border: `0.5px solid ${LINE2}`,
        borderRadius: 10,
        color: FG,
        fontFamily:
          "ui-sans-serif, -apple-system, BlinkMacSystemFont, system-ui, sans-serif",
        display: "flex",
        flexDirection: "column",
        pointerEvents: "auto",
        overflow: "hidden",
        zIndex: 5,
        transition: "width 0.25s ease",
      }}
    >
      <header
        style={{
          padding: "10px 12px",
          borderBottom: `0.5px solid ${LINE}`,
          display: "flex",
          justifyContent: collapsed ? "center" : "space-between",
          alignItems: "center",
          gap: 8,
        }}
      >
        {!collapsed && (
          <span
            style={{
              fontFamily: FONT_MONO,
              fontSize: 11,
              letterSpacing: "0.14em",
              color: FG3,
            }}
          >
            competitors
          </span>
        )}
        <button
          onClick={onToggleCollapse}
          style={collapseBtnStyle}
          aria-label={
            collapsed ? "Expand competitors" : "Collapse competitors"
          }
          title={collapsed ? "Expand" : "Collapse"}
        >
          {collapsed ? "‹" : "›"}
        </button>
      </header>

      {!collapsed && (
        <AnimatedLeaderboard
          visible={visible}
          snapshot={snapshot}
          incubating={incubating}
          stabilityGate={stabilityGate}
          symmetryGate={symmetryGate}
          leaderboardGraceSeconds={leaderboardGraceSeconds}
          onHover={onHover}
        />
      )}
    </aside>
  )
}

function OrganismRow({
  org,
  rank,
  onHover,
  isHighlighted,
  thumbCssWidth,
  thumbCssHeight,
}: {
  org: Organism
  rank: number
  onHover: (id: number | null) => void
  isHighlighted: boolean
  thumbCssWidth: number
  thumbCssHeight: number
}) {
  return (
    <div
      onMouseEnter={() => onHover(org.id)}
      onMouseLeave={() => onHover(null)}
      style={{
        padding: 10,
        border: `0.5px solid ${isHighlighted ? "rgba(93,202,165,0.45)" : LINE}`,
        background: isHighlighted ? "rgba(93,202,165,0.06)" : "transparent",
        borderRadius: 8,
        display: "flex",
        gap: 10,
        alignItems: "stretch",
        cursor: "default",
        transition: "border-color 0.15s ease, background 0.15s ease",
      }}
    >
      <Thumbnail
        org={org}
        thumbCssWidth={thumbCssWidth}
        thumbCssHeight={thumbCssHeight}
      />
      <div
        style={{
          flex: 1,
          minWidth: 0,
          display: "flex",
          flexDirection: "column",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            gap: 6,
            marginBottom: 2,
          }}
        >
          <span
            style={{
              color: FG3,
              fontFamily: FONT_MONO,
              fontSize: 11,
            }}
          >
            #{rank}
          </span>
          <span
            style={{
              fontSize: 13,
              fontWeight: 500,
              color: FG,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {org.name}
          </span>
        </div>
        <div
          style={{
            fontFamily: FONT_MONO,
            fontSize: 10,
            color: FG3,
            marginBottom: 5,
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
        >
          {org.signature}
        </div>
        <div
          style={{
            display: "flex",
            gap: 10,
            fontSize: 10.5,
            fontFamily: FONT_MONO,
            color: FG2,
          }}
        >
          <span title="size in particles">{org.size}p</span>
          <span title="age">{org.ageSeconds.toFixed(1)}s</span>
          <span title="speed">v{org.speedAvg.toFixed(2)}</span>
          <span title="distance traveled">d{org.distance.toFixed(2)}</span>
        </div>
        <StabilityBar value={org.leaderboardAvgComposite} />
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 6,
            rowGap: 2,
            marginTop: 4,
            fontFamily: FONT_MONO,
            fontSize: 9,
            color: FG4,
            justifyContent: "space-between",
          }}
        >
          <span title="rolling average composite stability (same window as leaderboard gate)">
            stab {(org.leaderboardAvgComposite * 100).toFixed(0)}%
          </span>
          <span title="velocity coherence">
            coh {(org.stability.velocityCoherence * 100).toFixed(0)}%
          </span>
          <span title="rolling average symmetry (listing gate)">
            sym {(org.leaderboardAvgSymmetry * 100).toFixed(0)}%
          </span>
          <span title="composite score">
            score {org.score.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  )
}

function Thumbnail({
  org,
  thumbCssWidth,
  thumbCssHeight,
}: {
  org: Organism
  thumbCssWidth: number
  thumbCssHeight: number
}) {
  const ref = useRef<HTMLCanvasElement | null>(null)
  const img = org.thumbnail
  useEffect(() => {
    if (!ref.current || !img) return
    const ctx = ref.current.getContext("2d")
    if (!ctx) return
    ctx.imageSmoothingEnabled = false
    ctx.putImageData(img, 0, 0)
  }, [img])
  return (
    <canvas
      ref={ref}
      width={img?.width ?? 1}
      height={img?.height ?? 1}
      style={{
        width: thumbCssWidth,
        height: thumbCssHeight,
        borderRadius: 6,
        background: "#0a0a0a",
        border: `0.5px solid ${LINE}`,
        flexShrink: 0,
        imageRendering: "pixelated",
      }}
    />
  )
}

function StabilityBar({ value }: { value: number }) {
  return (
    <div
      style={{
        marginTop: 6,
        height: 3,
        background: "rgba(255,255,255,0.08)",
        borderRadius: 2,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: `${Math.round(value * 100)}%`,
          height: "100%",
          background: ACCENT,
          transition: "width 0.4s ease",
        }}
      />
    </div>
  )
}

function DeadRow({ org }: { org: Organism }) {
  const cause = org.deathCause ?? "dissolution"
  const icon = cause === "absorbed" ? "⊕" : cause === "split" ? "⤴" : "✕"
  const causeColor =
    cause === "absorbed" ? "#FFD23B" : cause === "split" ? "#4AA8FF" : FG3
  return (
    <div
      style={{
        padding: "6px 8px",
        marginBottom: 3,
        borderRadius: 6,
        fontSize: 11,
        color: FG3,
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}
    >
      <span
        style={{
          fontFamily: FONT_MONO,
          width: 14,
          textAlign: "center",
          color: causeColor,
        }}
        aria-hidden="true"
      >
        {icon}
      </span>
      <span
        style={{
          flex: 1,
          color: FG2,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {org.name}
      </span>
      <span style={{ fontFamily: FONT_MONO, fontSize: 10, color: FG3 }}>
        {org.ageSeconds.toFixed(1)}s · {cause}
      </span>
    </div>
  )
}

const collapseBtnStyle: CSSProperties = {
  background: "transparent",
  border: `0.5px solid ${LINE2}`,
  color: FG2,
  width: 22,
  height: 22,
  borderRadius: 4,
  cursor: "pointer",
  fontFamily: FONT_MONO,
  fontSize: 14,
  lineHeight: 1,
  padding: 0,
}
