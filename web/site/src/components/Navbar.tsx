import React, { useState } from 'react'
import { Link, NavLink } from 'react-router-dom'

export default function Navbar() {
  const [open, setOpen] = useState(false)
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-2 rounded-md text-sm font-medium ${isActive ? 'bg-blue-600 text-white' : 'text-slate-700 hover:bg-slate-100'}`
  return (
    <nav className="border-b bg-white sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex h-14 items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <img src={new URL('../assets/logo.svg', import.meta.url).href} className="h-7 w-7" />
            <span className="font-semibold">StudyGlance</span>
          </Link>
          <div className="hidden md:flex gap-2">
            <NavLink to="/" className={linkClass}>Home</NavLink>
            <NavLink to="/download" className={linkClass}>Download</NavLink>
            <NavLink to="/quickstart" className={linkClass}>Quickstart</NavLink>
          </div>
          <button className="md:hidden p-2" aria-label="Menu" onClick={() => setOpen(!open)}>
            <span className="i">☰</span>
          </button>
        </div>
      </div>
      {open && (
        <div className="md:hidden border-t bg-white">
          <div className="px-4 py-2 space-y-1">
            <NavLink to="/" className={linkClass} onClick={() => setOpen(false)}>Home</NavLink>
            <NavLink to="/download" className={linkClass} onClick={() => setOpen(false)}>Download</NavLink>
            <NavLink to="/quickstart" className={linkClass} onClick={() => setOpen(false)}>Quickstart</NavLink>
          </div>
        </div>
      )}
    </nav>
  )
}

