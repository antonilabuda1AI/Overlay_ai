import React from 'react'
import { Link } from 'react-router-dom'

export default function Hero() {
  return (
    <section className="bg-gradient-to-br from-blue-600 to-cyan-500 text-white">
      <div className="max-w-6xl mx-auto px-4 py-20">
        <div className="max-w-3xl">
          <h1 className="text-4xl md:text-5xl font-bold leading-tight">Your On‑Screen Study Helper</h1>
          <p className="mt-4 text-lg text-white/90">StudyGlance reads the text you see on your screen in real time, saves the useful bits privately on your computer, and answers your questions with clear references. No recordings. No uploads of images.</p>
          <div className="mt-6 flex gap-3">
            <Link to="/download" className="px-5 py-2.5 bg-white text-slate-900 rounded-md font-medium">Download</Link>
            <Link to="/quickstart" className="px-5 py-2.5 bg-white/10 hover:bg-white/15 rounded-md font-medium">Quickstart</Link>
          </div>
        </div>
      </div>
    </section>
  )
}
