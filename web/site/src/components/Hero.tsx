import React from 'react'
import { Link } from 'react-router-dom'

export default function Hero() {
  return (
    <section className="bg-[#f5f7fb] dark:bg-slate-900 hero-bg">
      <div className="max-w-6xl mx-auto px-4 py-12 md:py-16">
        <div className="grid md:grid-cols-2 gap-8 items-center">
          <div className="max-w-xl">
            <h1 className="text-5xl md:text-6xl font-bold leading-tight text-slate-900 dark:text-slate-100">Study Faster. Stay Private.</h1>
            <p className="mt-4 text-lg muted">Real-time screen capture. Fast, citation-first answers — all processed on your device.</p>
            <div className="mt-6 flex flex-wrap gap-3">
              <Link to="/download" className="btn-primary focus-ring">Download for Windows (.exe)</Link>
              <Link to="/quickstart" className="btn-outline focus-ring">Quickstart</Link>
            </div>
            <div className="mt-2 text-xs muted">Free • Local • Private</div>
          </div>
          <div className="order-first md:order-none">
            <div className="card p-4 md:p-6 shadow-md">
              {/* OverlayAI mockup (polished app look, 16:9, Tailwind-only) */}
              <div className="relative" aria-hidden="true">
                <div className="relative aspect-video">
                  {/* Background sheet for depth (slight vertical offset) */}
                  <div className="absolute inset-0 -z-10 translate-y-3 scale-[0.98] blur-sm">
                    <div className="h-full rounded-2xl bg-white/80 shadow-xl shadow-[0_20px_40px_rgba(0,0,0,0.04)]" />
                  </div>

                  {/* Main desktop window (app) */}
                  <div className="relative h-full rounded-2xl bg-white shadow-[0_20px_40px_rgba(0,0,0,0.04)] border border-slate-200 p-4 md:p-6">
                    {/* Top app bar with input and controls */}
                    <div className="flex items-center gap-3">
                      <div className="flex-1 h-10 rounded-xl bg-white ring-1 ring-[#2563eb]/30 shadow-sm" />
                      <div className="h-3 w-3 rounded-full bg-slate-200" />
                      <div className="h-3 w-3 rounded-full bg-slate-200" />
                    </div>

                    {/* OCR region with grey text lines + clear blue highlight */}
                    <div className="relative mt-4 rounded-xl bg-white border border-slate-200 p-4 shadow-sm">
                      <div className="space-y-2">
                        <div className="h-2.5 bg-slate-200 rounded" />
                        <div className="h-2.5 bg-slate-200 rounded w-11/12" />
                        <div className="h-2.5 bg-slate-200 rounded w-10/12" />
                      </div>
                      <div className="mt-3 space-y-2">
                        <div className="h-2.5 bg-slate-200 rounded w-9/12" />
                        <div className="h-2.5 bg-slate-200 rounded w-8/12" />
                        <div className="h-2.5 bg-slate-200 rounded w-7/12" />
                      </div>
                      <div className="mt-3 space-y-2">
                        <div className="h-2.5 bg-slate-200 rounded" />
                        <div className="h-2.5 bg-slate-200 rounded w-5/6" />
                        <div className="h-2.5 bg-slate-200 rounded w-4/6" />
                      </div>

                      {/* OCR highlight overlay (text selection style: subtle vertical gradient + darker blue border) */}
                      <div className="pointer-events-none absolute left-3 right-3 top-10 h-16 rounded-lg bg-gradient-to-b from-[#e0f2ff] to-[#dbeafe] ring-1 ring-[#60a5fa] shadow-lg" />
                    </div>

                    {/* Floating answer bubble (overlay, stronger shadow, outside edge) */}
                    <div className="absolute -right-4 -bottom-4 w-64 rounded-2xl bg-white border border-slate-200 shadow-2xl shadow-[0_24px_48px_rgba(0,0,0,0.12)] p-3">
                      {/* Blue AI dot */}
                      <div className="absolute left-3 top-3 h-2.5 w-2.5 rounded-full bg-[#2563eb]" />
                      {/* Answer highlight (single stronger blue line) */}
                      <div className="ml-6 h-1.5 bg-[#2563eb] rounded mb-2" />
                      <div className="ml-6 space-y-1.5">
                        <div className="h-2.5 bg-slate-200 rounded w-3/4" />
                        <div className="h-2.5 bg-slate-200 rounded w-2/3" />
                        <div className="h-2.5 bg-slate-200 rounded w-1/2" />
                      </div>
                      {/* Bubble tail */}
                      <div className="absolute -bottom-2 right-10 h-3 w-3 bg-white border border-slate-200 rotate-45 shadow-sm" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
