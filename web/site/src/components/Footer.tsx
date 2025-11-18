import React from 'react'
import { Link } from 'react-router-dom'

export default function Footer() {
  return (
    <footer className="mt-16 border-t bg-white">
      <div className="max-w-6xl mx-auto px-4 py-6 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-slate-600">
          <img src={new URL('../assets/logo.svg', import.meta.url).href} className="h-6 w-6" />
          <span>© {new Date().getFullYear()} StudyGlance</span>
        </div>
        <div className="flex gap-4 text-sm">
          <Link to="/download" className="text-slate-600 hover:text-slate-900">Download</Link>
          <Link to="/quickstart" className="text-slate-600 hover:text-slate-900">Quickstart</Link>
          <a href="#" className="text-slate-600 hover:text-slate-900">Privacy</a>
        </div>
      </div>
    </footer>
  )
}

