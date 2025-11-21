import React from 'react'
import Hero from '../components/Hero'
import FeatureGrid from '../components/FeatureGrid'

export default function Home() {
  return (
    <main>
      <Hero />
      <FeatureGrid />
      <section className="max-w-6xl mx-auto px-4 py-12">
        <h2 className="text-2xl font-semibold mb-4">How it works</h2>
        <ol className="grid gap-4 md:grid-cols-4 text-slate-700 dark:text-slate-300">
          <li className="card p-4">
            <div className="flex items-center gap-3 mb-2">
              <div className="h-8 w-8 rounded-full flex items-center justify-center text-white" style={{background:'var(--primary)'}}>1</div>
              <div className="font-medium">Start Live</div>
            </div>
            <p className="muted text-sm">Press Live. OverlayAI watches for small text changes on your screen.</p>
          </li>
          <li className="card p-4">
            <div className="flex items-center gap-3 mb-2">
              <div className="h-8 w-8 rounded-full flex items-center justify-center text-white" style={{background:'var(--primary)'}}>2</div>
              <div className="font-medium">Capture Text</div>
            </div>
            <p className="muted text-sm">It reads just that text (not the whole screen) and saves it to a timeline on your computer.</p>
          </li>
          <li className="card p-4">
            <div className="flex items-center gap-3 mb-2">
              <div className="h-8 w-8 rounded-full flex items-center justify-center text-white" style={{background:'var(--primary)'}}>3</div>
              <div className="font-medium">Ask & Cite</div>
            </div>
            <p className="muted text-sm">Type a question. You get a friendly answer and the exact lines it came from.</p>
          </li>
          <li className="card p-4">
            <div className="flex items-center gap-3 mb-2">
              <div className="h-8 w-8 rounded-full flex items-center justify-center text-white" style={{background:'var(--primary)'}}>4</div>
              <div className="font-medium">Stop Anytime</div>
            </div>
            <p className="muted text-sm">Stop Live any time. Nothing is recorded in the background.</p>
          </li>
        </ol>
      </section>
      <section className="max-w-6xl mx-auto px-4 pb-16">
        <div className="p-6 rounded-xl border" style={{background:'linear-gradient(0deg, rgba(41,110,255,0.06), rgba(41,110,255,0.06)), var(--card-bg)', borderColor:'var(--border-soft)'}}>
          <div className="flex items-start gap-3">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-[color:var(--primary)] mt-1">
              <path d="M12 3l7 4v5c0 5-3.5 8-7 9-3.5-1-7-4-7-9V7l7-4z" stroke="currentColor" strokeWidth="2"/>
              <path d="M9.5 12.5l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <div>
              <h3 className="text-xl font-semibold">Privacy promise</h3>
              <p className="mt-2 muted">OverlayAI never saves or uploads pictures of your screen. It only keeps the text it reads — on your device — and only sends that text to OpenAI when you ask a question.</p>
            </div>
          </div>
        </div>
        <div className="mt-8 flex gap-3">
          <a href="/download" className="btn-primary focus-ring">Get the app</a>
          <a href="/quickstart" className="btn-outline focus-ring">Read quickstart</a>
        </div>
      </section>
    </main>
  )
}
