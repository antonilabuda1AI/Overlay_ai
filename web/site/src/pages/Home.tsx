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
        <ol className="list-decimal ml-6 space-y-2 text-slate-700">
          <li>Press Live. StudyGlance watches for small text changes on your screen.</li>
          <li>It reads just that text (not the whole screen) and saves it to a timeline on your computer.</li>
          <li>Type a question. You get a friendly answer and the exact lines it came from.</li>
          <li>Stop Live any time. Nothing is recorded in the background.</li>
        </ol>
      </section>
      <section className="max-w-6xl mx-auto px-4 pb-16">
        <div className="rounded-lg border bg-white p-6">
          <h3 className="text-xl font-semibold">Privacy promise</h3>
          <p className="mt-2 text-slate-700">StudyGlance never saves or uploads pictures of your screen. It only keeps the text it reads — on your device — and only sends that text to OpenAI when you ask a question.</p>
        </div>
      </section>
    </main>
  )
}
