/**
 * Competitors HUD — right-side panel listing alive organisms (top N by score)
 * and a recently-dead feed. Hover a row to highlight on the main canvas.
 *
 * Visual aesthetic mirrors the Live View skin: monospace metadata, accent
 * green status dots, restrained borders. Dark glass background.
 */

import { useEffect, useRef } from "react"
import type { Organism, TrackerSnapshot } from "./tracker"

type Props = {
  snapshot: TrackerSnapshot
  stabilityGate: number
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

export function Competitors({
  snapshot,
  stabilityGate,
  topN,
  onHover,
  collapsed,
  onToggleCollapse,
}: Props) {
  const visible = snapshot.alive
    .filter((o) => o.stability.composite >= stabilityGate)
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
        fontFamily: "ui-sans-serif, -apple-system, BlinkMacSystemFont, system-ui, sans-serif",
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
          aria-label={collapsed ? "Expand competitors" : "Collapse competitors"}
          title={collapsed ? "Expand" : "Collapse"}
        >
          {collapsed ? "‹" : "›"}
        </button>
      </header>

      {!collapsed && (
        <div
          style={{
            flex: 1,
            overflowY: "auto",
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

          {visible.map((org, i) => (
            <OrganismRow
              key={org.id}
              org={org}
              rank={i + 1}
              onHover={onHover}
              isHighlighted={snapshot.highlightId === org.id}
            />
          ))}

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
              {incubating} incubating · stability gate {(stabilityGate * 100).toFixed(0)}%
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
      )}
    </aside>
  )
}

function OrganismRow({
  org,
  rank,
  onHover,
  isHighlighted,
}: {
  org: Organism
  rank: number
  onHover: (id: number | null) => void
  isHighlighted: boolean
}) {
  return (
    <div
      onMouseEnter={() => onHover(org.id)}
      onMouseLeave={() => onHover(null)}
      style={{
        padding: 10,
        marginBottom: 6,
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
      <Thumbnail org={org} />
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
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
        <StabilityBar value={org.stability.composite} />
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginTop: 4,
            fontFamily: FONT_MONO,
            fontSize: 9.5,
            color: FG4,
          }}
        >
          <span title="composite stability">stab {(org.stability.composite * 100).toFixed(0)}%</span>
          <span title="velocity coherence">coh {(org.stability.velocityCoherence * 100).toFixed(0)}%</span>
          <span title="composite score">score {org.score.toFixed(2)}</span>
        </div>
      </div>
    </div>
  )
}

function Thumbnail({ org }: { org: Organism }) {
  const ref = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    if (!ref.current || !org.thumbnail) return
    const ctx = ref.current.getContext("2d")
    if (!ctx) return
    ctx.imageSmoothingEnabled = false
    ctx.putImageData(org.thumbnail, 0, 0)
  }, [org.thumbnail])
  return (
    <canvas
      ref={ref}
      width={48}
      height={48}
      style={{
        width: 48,
        height: 48,
        borderRadius: 6,
        background: "#0a0a0a",
        border: `0.5px solid ${LINE}`,
        flexShrink: 0,
        // Crisp pixels on hi-DPI: keep the chunky aesthetic of the main sim.
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
      <span style={{ flex: 1, color: FG2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {org.name}
      </span>
      <span style={{ fontFamily: FONT_MONO, fontSize: 10, color: FG3 }}>
        {org.ageSeconds.toFixed(1)}s · {cause}
      </span>
    </div>
  )
}

const collapseBtnStyle: React.CSSProperties = {
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

