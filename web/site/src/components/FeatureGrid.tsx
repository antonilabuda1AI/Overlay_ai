import React from 'react'

const features = [
  { title: 'Simple Overlay', desc: 'Start/stop with a click or hotkeys. No clutter.', icon: '🎛️' },
  { title: 'Reads Text on Screen', desc: 'Grabs small changes only — never records video.', icon: '👀' },
  { title: 'Private by Design', desc: 'Everything stays on your device. Only text is used for answers.', icon: '🛡️' },
  { title: 'Answers with Proof', desc: 'Every answer links to the exact lines it used.', icon: '📎' },
  { title: 'Fast and Light', desc: 'Built to use little CPU and memory while you work.', icon: '⚡' },
  { title: 'Works Everywhere', desc: 'Windows, macOS, and Linux.', icon: '🖥️' },
]

export default function FeatureGrid() {
  return (
    <section className="bg-white">
      <div className="max-w-6xl mx-auto px-4 py-12">
        <h2 className="text-2xl font-semibold mb-6">Why people like StudyGlance</h2>
        <div className="grid md:grid-cols-3 gap-6">
          {features.map((f) => (
            <div key={f.title} className="rounded-lg border bg-white p-5 shadow-sm hover:shadow transition">
              <div className="text-2xl">{f.icon}</div>
              <div className="mt-2 font-medium">{f.title}</div>
              <p className="text-slate-600 text-sm mt-1">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
