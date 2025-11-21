import React from 'react'
import { Link } from 'react-router-dom'

export default function Footer() {
  return (
    <footer className="mt-16 bg-slate-950 text-slate-300">
      <div className="max-w-6xl mx-auto px-4 py-8 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <img src={new URL('../assets/logo.svg', import.meta.url).href} className="h-6 w-6" alt="OverlayAI" />
          <span>© 2025 OverlayAI</span>
        </div>
        <div className="flex gap-6 text-sm">
          <Link to="/download" className="hover:text-white">Download</Link>
          <Link to="/quickstart" className="hover:text-white">Quickstart</Link>
          <a href="#" className="hover:text-white">Privacy</a>
          <a href="https://github.com/" target="_blank" rel="noreferrer" className="hover:text-white">GitHub</a>
        </div>
      </div>
    </footer>
  )
}
