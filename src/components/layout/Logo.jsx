export default function Logo({ size = 32 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="CapScout logo"
    >
      {/* Magnifying glass circle — teal ring with subtle gradient fill */}
      <defs>
        <linearGradient id="glass-bg" x1="10" y1="10" x2="46" y2="46" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#e6fcf9" />
          <stop offset="100%" stopColor="#eef2ff" />
        </linearGradient>
        <linearGradient id="handle-grad" x1="41" y1="41" x2="56" y2="56" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#0d9488" />
          <stop offset="100%" stopColor="#4f46e5" />
        </linearGradient>
      </defs>

      <circle cx="28" cy="28" r="18" stroke="#0d9488" strokeWidth="4" fill="url(#glass-bg)" />

      {/* Outer accent arc — indigo, top-right quarter for depth */}
      <path
        d="M40.7 15.3A18 18 0 0 1 44 22"
        stroke="#4f46e5"
        strokeWidth="3.5"
        strokeLinecap="round"
        fill="none"
      />

      {/* House roof — burnt orange */}
      <path d="M28 16L16 26.5H40L28 16Z" fill="#BF5700" />

      {/* House body — teal */}
      <rect x="19" y="26" width="18" height="12" rx="1" fill="#0d9488" />

      {/* Door — indigo */}
      <rect x="25" y="30" width="6" height="8" rx="1" fill="#4f46e5" />

      {/* Windows — bright white with orange tint */}
      <rect x="20.5" y="28" width="4" height="3.5" rx="0.7" fill="#FFF3E6" />
      <rect x="32.5" y="28" width="4" height="3.5" rx="0.7" fill="#FFF3E6" />

      {/* Chimney — orange */}
      <rect x="33" y="18" width="3" height="7" rx="0.8" fill="#BF5700" />

      {/* Magnifying glass handle — gradient teal→indigo */}
      <line
        x1="41" y1="41"
        x2="56" y2="56"
        stroke="url(#handle-grad)"
        strokeWidth="5"
        strokeLinecap="round"
      />

      {/* Handle highlight */}
      <line
        x1="42.5" y1="42.5"
        x2="54" y2="54"
        stroke="#6366f1"
        strokeWidth="1.8"
        strokeLinecap="round"
        opacity="0.5"
      />
    </svg>
  );
}
