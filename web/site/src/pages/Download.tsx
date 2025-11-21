import React from 'react'

export default function Download() {
  const cards = [
    { os: 'Windows', ext: '.exe', note: 'Installer', href: '#' },
    { os: 'macOS', ext: '.dmg', note: 'App + DMG', href: '#' },
    { os: 'Linux', ext: 'AppImage', note: 'Portable', href: '#' },
  ]
  return (
    <main className="max-w-6xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-bold mb-6">Download OverlayAI</h1>
      <div className="grid md:grid-cols-3 gap-6">
        {cards.map(c => (
          <a
            key={c.os}
            href={c.href}
            className={`card p-6 transition block focus-ring ${c.os==='Windows' ? 'shadow-md border-2' : 'hover:shadow-md'}`}
            style={c.os==='Windows' ? {borderColor:'var(--primary)'} : {}}
          >
            <div className="muted text-sm">{c.note}</div>
            <div className="mt-1 text-lg font-semibold">{c.os} <span className="text-slate-400">{c.ext}</span></div>
            {c.os==='Windows' ? (
              <div className="mt-4"><span className="btn-primary focus-ring">Download</span></div>
            ) : (
              <div className="mt-3"><span className="badge">Coming soon</span></div>
            )}
          </a>
        ))}
      </div>
      <div className="mt-8 rounded-md bg-slate-100 dark:bg-slate-800 p-4 text-sm muted">
        We’ll publish download links and checksums here when releases are ready.
      </div>
    </main>
  )
}
