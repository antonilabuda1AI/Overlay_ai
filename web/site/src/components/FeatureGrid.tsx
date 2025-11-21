import React from 'react'

const IconWindow = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" className="text-[color:var(--primary)]">
    <rect x="3" y="5" width="18" height="14" rx="2" className="stroke-current" strokeWidth="2" />
    <path d="M3 9h18" className="stroke-current" strokeWidth="2" />
  </svg>
)
const IconSearch = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" className="text-[color:var(--primary)]">
    <circle cx="11" cy="11" r="7" className="stroke-current" strokeWidth="2" />
    <path d="M20 20l-3-3" className="stroke-current" strokeWidth="2" strokeLinecap="round" />
  </svg>
)
const IconLock = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" className="text-[color:var(--primary)]">
    <rect x="4" y="10" width="16" height="10" rx="2" className="stroke-current" strokeWidth="2" />
    <path d="M8 10V8a4 4 0 018 0v2" className="stroke-current" strokeWidth="2" />
  </svg>
)
const IconBrain = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" className="text-[color:var(--primary)]">
    <path d="M8 6a3 3 0 013-3 3 3 0 013 3v12a3 3 0 11-6 0V6z" className="stroke-current" strokeWidth="2" />
    <path d="M14 6a3 3 0 016 0v7a3 3 0 11-6 0V6zM4 9a3 3 0 016 0v7a3 3 0 11-6 0V9z" className="stroke-current" strokeWidth="2" />
  </svg>
)
const IconBolt = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" className="text-[color:var(--primary)]">
    <path d="M13 2L3 14h7l-1 8 10-12h-7l1-8z" className="stroke-current" strokeWidth="2" fill="none" />
  </svg>
)
const IconDesktop = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" className="text-[color:var(--primary)]">
    <rect x="3" y="4" width="18" height="12" rx="2" className="stroke-current" strokeWidth="2" />
    <path d="M8 20h8" className="stroke-current" strokeWidth="2" strokeLinecap="round" />
  </svg>
)

const features = [
  { title: 'Simple Overlay', desc: 'Start or stop with a click or hotkey. No clutter.', icon: <IconWindow /> },
  { title: 'Reads Text on Screen', desc: 'Captures small text changes only — never records video.', icon: <IconSearch /> },
  { title: 'Private by Design', desc: 'Everything stays on your device. Only text is used for answers.', icon: <IconLock /> },
  { title: 'Answers with Citations', desc: 'Every answer links to the exact lines it used.', icon: <IconBrain /> },
  { title: 'Fast and Light', desc: 'Built to use little CPU and memory while you work.', icon: <IconBolt /> },
  { title: 'Cross‑platform', desc: 'Works on Windows, macOS, and Linux.', icon: <IconDesktop /> },
]

export default function FeatureGrid() {
  return (
    <section className="bg-white dark:bg-slate-900">
      <div className="max-w-6xl mx-auto px-4 py-12">
        <h2 className="text-2xl font-semibold mb-6">Why people like OverlayAI</h2>
        <div className="grid md:grid-cols-3 gap-6">
          {features.map((f) => (
            <div key={f.title} className="card p-6 hover:shadow-md transition shadow-sm hover:-translate-y-0.5 h-full flex flex-col">
              <div className="text-3xl mb-2">{f.icon}</div>
              <div className="mt-2 font-medium">{f.title}</div>
              <p className="muted text-sm mt-1">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
