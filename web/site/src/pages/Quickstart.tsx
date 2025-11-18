import React, { useState } from 'react'

type Tab = 'Windows' | 'macOS' | 'Linux'

export default function Quickstart() {
  const [tab, setTab] = useState<Tab>('Windows')
  const cmd = (s: string) => <code className="bg-slate-100 px-2 py-1 rounded text-sm">{s}</code>
  return (
    <main className="max-w-6xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-bold mb-4">Quickstart</h1>
      <p className="text-slate-700 mb-6">Follow these quick steps to get started on your computer.</p>
      <div className="inline-flex rounded-md border bg-white overflow-hidden">
        {(['Windows','macOS','Linux'] as Tab[]).map(t => (
          <button key={t} onClick={() => setTab(t)} className={`px-4 py-2 text-sm ${tab===t?'bg-blue-600 text-white':'hover:bg-slate-100'}`}>{t}</button>
        ))}
      </div>
      <div className="mt-6 rounded-lg border bg-white p-5">
        <ol className="list-decimal ml-6 space-y-2">
          <li>Install Tesseract (the text reader).</li>
          <li>Set your OpenAI key: {cmd('OPENAI_API_KEY')}</li>
          <li>Start the backend: {cmd('cd backend && uvicorn app.main:app --reload')}</li>
          <li>Start the overlay: {cmd('cd apps/overlay/pyqt && python main.py')}</li>
          <li>Open the website (optional): {cmd('cd web/site && npm install && npm run dev')}</li>
        </ol>
      </div>
    </main>
  )}
