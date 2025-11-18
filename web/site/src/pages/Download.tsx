import React from 'react'

export default function Download() {
  const cards = [
    { os: 'Windows', ext: '.exe', note: 'Installer', href: '#'},
    { os: 'macOS', ext: '.dmg', note: 'App + DMG', href: '#'},
    { os: 'Linux', ext: 'AppImage', note: 'Portable', href: '#'},
  ]
  return (
    <main className="max-w-6xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-bold mb-6">Download StudyGlance</h1>
      <div className="grid md:grid-cols-3 gap-6">
        {cards.map(c => (
          <a key={c.os} href={c.href} className="rounded-lg border bg-white p-5 shadow-sm hover:shadow transition block">
            <div className="text-slate-500 text-sm">{c.note}</div>
            <div className="mt-1 text-lg font-semibold">{c.os} <span className="text-slate-400">{c.ext}</span></div>
            <div className="mt-3 text-blue-600">Coming soon</div>
          </a>
        ))}
      </div>
      <div className="mt-8 rounded-md bg-slate-100 p-4 text-slate-700 text-sm">
        We’ll share download links and checksums here when releases are ready.
      </div>
    </main>
  )
}
