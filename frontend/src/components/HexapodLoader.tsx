type HexapodLoaderProps = {
  exiting?: boolean
  label?: string
  ready?: boolean
}

type LegSpec = {
  shoulder: [number, number]
  knee: [number, number]
  foot: [number, number]
  group: 'a' | 'b'
}

const LEGS: LegSpec[] = [
  { shoulder: [84, 64], knee: [-14, -6], foot: [-26, -18], group: 'a' }, // FL
  { shoulder: [116, 64], knee: [14, -6], foot: [26, -18], group: 'b' }, // FR
  { shoulder: [78, 80], knee: [-16, 0], foot: [-30, 2], group: 'b' }, // ML
  { shoulder: [122, 80], knee: [16, 0], foot: [30, 2], group: 'a' }, // MR
  { shoulder: [84, 96], knee: [-14, 6], foot: [-26, 18], group: 'a' }, // BL
  { shoulder: [116, 96], knee: [14, 6], foot: [26, 18], group: 'b' }, // BR
]

export function HexapodLoader({
  exiting = false,
  label = 'Loading Hexy',
  ready = false,
}: HexapodLoaderProps) {
  return (
    <div
      className="hexy-loader"
      data-exiting={exiting ? 'true' : 'false'}
      data-ready={ready ? 'true' : 'false'}
      role="status"
      aria-live="polite"
      aria-busy={!ready}
    >
      <svg
        viewBox="0 0 200 160"
        width="200"
        height="160"
        aria-hidden="true"
        focusable="false"
        stroke="currentColor"
        strokeWidth={3}
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      >
        <g className="hexy-loader-body">
          {LEGS.map(({ shoulder, knee, foot, group }, i) => {
            const [sx, sy] = shoulder
            const kx = sx + knee[0]
            const ky = sy + knee[1]
            const fx = sx + foot[0]
            const fy = sy + foot[1]
            return (
              <g
                key={i}
                className={`hexy-loader-leg hexy-loader-leg-${group}`}
                style={{ transformOrigin: `${sx}px ${sy}px` }}
              >
                <path d={`M${sx},${sy} L${kx},${ky} L${fx},${fy}`} />
                <circle cx={sx} cy={sy} r={3} fill="currentColor" stroke="none" />
                <circle cx={fx} cy={fy} r={2} fill="currentColor" stroke="none" />
              </g>
            )
          })}

          <g transform="rotate(45 100 80)">
            <rect
              x={88}
              y={68}
              width={24}
              height={24}
              rx={2}
              fill="currentColor"
              stroke="none"
            />
            <rect
              x={92}
              y={72}
              width={16}
              height={16}
              rx={1.5}
              fill="none"
              stroke="rgba(41,41,41,0.55)"
              strokeWidth={1.25}
            />
          </g>

          <path
            d="M96,64 L100,60 L104,64"
            stroke="currentColor"
            strokeWidth={1.75}
            fill="none"
          />
        </g>
      </svg>
      <span className="hexy-loader-label">{label}</span>
    </div>
  )
}

export default HexapodLoader
