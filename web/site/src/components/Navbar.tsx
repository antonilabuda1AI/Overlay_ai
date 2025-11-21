import React, { useEffect, useState } from 'react'
import { Link, NavLink } from 'react-router-dom'

export default function Navbar() {
  const [open, setOpen] = useState(false)
  const [dark, setDark] = useState(false)

  useEffect(() => {
    try {
      const saved = localStorage.getItem('theme')
      const d = saved ? saved === 'dark' : document.documentElement.classList.contains('dark')
      setDark(d)
    } catch {}
  }, [])

  useEffect(() => {
    if (dark) {
      document.documentElement.classList.add('dark')
      try { localStorage.setItem('theme', 'dark') } catch {}
    } else {
      document.documentElement.classList.remove('dark')
      try { localStorage.setItem('theme', 'light') } catch {}
    }
  }, [dark])

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-2 text-sm font-medium rounded-md focus-ring relative ${
      isActive
        ? 'text-slate-900 dark:text-blue-400 after:absolute after:left-2 after:right-2 after:-bottom-1 after:h-0.5 after:bg-[color:var(--primary)]'
        : 'text-slate-900 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-800'
    }`

  return (
    <nav className="border-b border-slate-300 dark:border-slate-800 bg-white dark:bg-slate-900 sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex h-14 items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <img src={new URL('../assets/logo.svg', import.meta.url).href} className="h-7 w-7" alt="OverlayAI" />
            <span className="font-semibold">OverlayAI</span>
          </Link>
          <div className="hidden md:flex items-center gap-2">
            <NavLink to="/" className={linkClass}>Home</NavLink>
            <NavLink to="/quickstart" className={linkClass}>Quickstart</NavLink>
          </div>
          <div className="hidden md:flex items-center gap-2">
            <button aria-label="Toggle theme" onClick={() => setDark(d => !d)} className="btn-outline focus-ring px-2">
              <span className="sr-only">Toggle theme</span>
              <span className="inline-flex items-center gap-2 text-sm">
                <span className="text-slate-600 dark:text-slate-300">{dark ? 'Dark' : 'Light'}</span>
                <span className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${dark ? 'bg-[color:var(--primary)]' : 'bg-slate-300'}`}>
                  <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${dark ? 'translate-x-4' : 'translate-x-1'}`} />
                </span>
              </span>
            </button>
            <Link to="/download" className="btn-primary focus-ring">Download</Link>
          </div>
          <button className="md:hidden p-2 rounded hover:bg-slate-100 dark:hover:bg-slate-800" aria-label="Menu" onClick={() => setOpen(!open)}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
              <path fillRule="evenodd" d="M3 6.75A.75.75 0 013.75 6h16.5a.75.75 0 010 1.5H3.75A.75.75 0 013 6.75zm0 5.25a.75.75 0 01.75-.75h16.5a.75.75 0 010 1.5H3.75A.75.75 0 013 12zm.75 4.5a.75.75 0 000 1.5h16.5a.75.75 0 000-1.5H3.75z" clipRule="evenodd" />
            </svg>
          </button>
        </div>
      </div>
      {open && (
        <div className="md:hidden border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
          <div className="px-4 py-2 space-y-1">
            <NavLink to="/" className={linkClass} onClick={() => setOpen(false)}>Home</NavLink>
            <NavLink to="/quickstart" className={linkClass} onClick={() => setOpen(false)}>Quickstart</NavLink>
            <div className="flex gap-2 pt-2">
              <button
                aria-label="Toggle theme"
                onClick={() => setDark(d => !d)}
                className="btn-outline focus-ring flex-1 text-left"
              >
                {dark ? 'Light' : 'Dark'}
              </button>
              <Link to="/download" onClick={() => setOpen(false)} className="btn-primary focus-ring">Download</Link>
            </div>
          </div>
        </div>
      )}
    </nav>
  )
}
