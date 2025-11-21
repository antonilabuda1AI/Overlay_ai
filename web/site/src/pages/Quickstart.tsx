import React, { useState } from 'react'

type Tab = 'Windows' | 'macOS' | 'Linux'

export default function Quickstart() {
  const [tab, setTab] = useState<Tab>('Windows')
  const [showAdv, setShowAdv] = useState(false)

  const Code = (s: string) => (
    <code className="inline-block bg-slate-50 border border-slate-200 text-slate-800 dark:bg-slate-800 dark:text-slate-100 dark:border-slate-700 rounded-lg px-2.5 py-1 font-mono text-[13px]">
      {s}
    </code>
  )

  const Tabs = () => (
    <div className="inline-flex rounded-xl border border-slate-200 bg-white dark:bg-slate-900 overflow-hidden">
      {(['Windows', 'macOS', 'Linux'] as Tab[]).map((t) => (
        <button
          key={t}
          onClick={() => setTab(t)}
          className={`px-4 py-2 text-sm focus-ring relative ${
            tab === t
              ? 'bg-white dark:bg-slate-900 font-medium text-slate-900 dark:text-slate-100 ring-2 ring-[color:var(--primary)] z-10'
              : 'text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800'
          }`}
        >
          {t}
        </button>
      ))}
    </div>
  )

  const StepNum = ({ n }: { n: number }) => (
    <div className="flex flex-col items-center">
      <div className="h-8 w-8 rounded-full bg-[color:var(--primary)] text-white flex items-center justify-center text-sm font-semibold shadow-sm">
        {n}
      </div>
      <div className="w-px flex-1 bg-slate-200" />
    </div>
  )

  const Step = ({ n, icon, title, children }: { n: number; icon: React.ReactNode; title: string; children?: React.ReactNode }) => (
    <div className="grid grid-cols-[auto,1fr] gap-4 items-start">
      <StepNum n={n} />
      <div className="rounded-xl border border-slate-200 bg-white dark:bg-slate-900 dark:border-slate-800 shadow-sm p-4">
        <div className="flex items-center gap-2 mb-2">
          <div className="text-xl" aria-hidden="true">{icon}</div>
          <div className="font-medium text-slate-900 dark:text-slate-100">{title}</div>
        </div>
        {children && <div className="text-sm text-slate-600 dark:text-slate-300">{children}</div>}
      </div>
    </div>
  )

  const WindowsSteps = () => (
    <div className="space-y-5">
      <Step n={1} icon={<span>📥</span>} title="Download OverlayAI.exe">
        Get it from the Download page.
      </Step>
      <Step n={2} icon={<span>🧩</span>} title="Run the installer">
        Follow the prompts. OverlayAI installs to your system.
      </Step>
      <Step n={3} icon={<span>🚀</span>} title="Launch the app">
        Start OverlayAI from the Start Menu or desktop shortcut.
      </Step>
    </div>
  )

  const MacSteps = () => (
    <div className="space-y-5">
      <Step n={1} icon={<span>📥</span>} title="Download OverlayAI.dmg">
        From the Download page (coming soon).
      </Step>
      <Step n={2} icon={<span>📦</span>} title="Open the DMG and drag to Applications">
        Move OverlayAI into Applications.
      </Step>
      <Step n={3} icon={<span>🚀</span>} title="Launch the app">
        Open from Applications. Grant screen capture permission if prompted.
      </Step>
    </div>
  )

  const LinuxSteps = () => (
    <div className="space-y-5">
      <Step n={1} icon={<span>📥</span>} title="Download OverlayAI.AppImage">
        From the Download page (coming soon). Make it executable.
      </Step>
      <Step n={2} icon={<span>🔧</span>} title="Run the AppImage">
        Double‑click or run {Code('./OverlayAI.AppImage')} in a terminal.
      </Step>
      <Step n={3} icon={<span>🚀</span>} title="Launch the app">
        Grant screen capture permissions if needed.
      </Step>
    </div>
  )

  const AdvancedPanel = () => (
    <div className="mt-8">
      <button
        onClick={() => setShowAdv((s) => !s)}
        className="w-full flex items-center justify-between rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 shadow-sm hover:bg-slate-50 dark:hover:bg-slate-800 focus-ring"
      >
        <div className="flex items-center gap-2">
          <span className="text-slate-900 dark:text-slate-100 font-medium">Advanced Setup (Developers)</span>
          <span className="text-xs text-slate-500">Backend / Python / uvicorn</span>
        </div>
        <svg className={`h-5 w-5 text-slate-500 transition-transform ${showAdv ? 'rotate-180' : ''}`} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
          <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 10.94l3.71-3.71a.75.75 0 111.06 1.06l-4.24 4.24a.75.75 0 01-1.06 0L5.21 8.29a.75.75 0 01.02-1.08z" clipRule="evenodd" />
        </svg>
      </button>
      {showAdv && (
        <div className="mt-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 shadow-sm space-y-4">
          <div>
            <div className="font-medium mb-1">Prereqs</div>
            <ul className="list-disc ml-5 text-sm text-slate-600 dark:text-slate-300">
              <li>Python 3.10+</li>
              <li>Tesseract OCR</li>
              <li>Node.js 18+ (for website preview)</li>
            </ul>
          </div>
          <div>
            <div className="font-medium mb-2">Environment</div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800 p-3">
              <div className="text-sm text-slate-700 dark:text-slate-200 space-y-2 overflow-x-auto">
                <div>{Code('setx OPENAI_API_KEY "<your-key>"    # Windows')}</div>
                <div>{Code('export OPENAI_API_KEY="<your-key>"   # macOS/Linux')}</div>
              </div>
            </div>
          </div>
          <div>
            <div className="font-medium mb-2">Run backend</div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800 p-3">
              <div className="text-sm text-slate-700 dark:text-slate-200 space-y-2 overflow-x-auto">
                <div>{Code('cd backend')}</div>
                <div>{Code('uvicorn app.main:app --reload --port 8000')}</div>
              </div>
            </div>
          </div>
          <div>
            <div className="font-medium mb-2">Run overlay (desktop)</div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800 p-3">
              <div className="text-sm text-slate-700 dark:text-slate-200 space-y-2 overflow-x-auto">
                <div>{Code('cd apps/overlay/pyqt')}</div>
                <div>{Code('python main.py')}</div>
              </div>
            </div>
          </div>
          <div>
            <div className="font-medium mb-2">Preview website (optional)</div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800 p-3">
              <div className="text-sm text-slate-700 dark:text-slate-200 space-y-2 overflow-x-auto">
                <div>{Code('cd web/site')}</div>
                <div>{Code('npm install')}</div>
                <div>{Code('npm run dev')}</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )

  return (
    <main className="max-w-6xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-bold mb-2">Quickstart</h1>
      <p className="text-slate-700 dark:text-slate-300 mb-6">Install OverlayAI in a few simple steps. Developers can expand advanced setup below.</p>

      <Tabs />

      <div className="mt-6 card p-6 md:p-7 space-y-6">
        {tab === 'Windows' && <WindowsSteps />}
        {tab === 'macOS' && <MacSteps />}
        {tab === 'Linux' && <LinuxSteps />}
      </div>

      <AdvancedPanel />
    </main>
  )
}
