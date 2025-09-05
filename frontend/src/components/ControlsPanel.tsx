\
import { useState } from 'react'

export type Params = {
  perplexity: number
  n_iter: number
  learning_rate: 'auto' | number
  nu: number
  gamma: 'scale' | 'auto' | number
  kernel: 'rbf' | 'linear' | 'poly' | 'sigmoid'
  random_state: number | null
}

export default function ControlsPanel({
  onRun
}: {
  onRun: (p: Params) => void
}) {
  const [perplexity, setPerplexity] = useState(30)
  const [nIter, setNIter] = useState(1000)
  const [nu, setNu] = useState(0.1)
  const [gamma, setGamma] = useState<'scale' | 'auto'>('scale')
  const [kernel, setKernel] = useState<'rbf' | 'linear' | 'poly' | 'sigmoid'>('rbf')
  const [busy, setBusy] = useState(false)

  return (
    <div className="grid grid-cols-1 gap-4 rounded-xl border border-slate-800 bg-slate-900/40 p-4 md:grid-cols-3">
      <div>
        <label className="block text-xs text-slate-400">Perplexity: {perplexity}</label>
        <input type="range" min={5} max={80} value={perplexity} onChange={(e) => setPerplexity(parseInt(e.target.value))} className="w-full" />
      </div>
      <div>
        <label className="block text-xs text-slate-400">t‑SNE Iterations: {nIter}</label>
        <input type="range" min={250} max={4000} step={250} value={nIter} onChange={(e) => setNIter(parseInt(e.target.value))} className="w-full" />
      </div>
      <div>
        <label className="block text-xs text-slate-400">OC‑SVM ν: {nu.toFixed(2)}</label>
        <input type="range" min={0.01} max={0.5} step={0.01} value={nu} onChange={(e) => setNu(parseFloat(e.target.value))} className="w-full" />
      </div>
      <div>
        <label className="block text-xs text-slate-400">OC‑SVM gamma</label>
        <select value={gamma} onChange={e => setGamma(e.target.value as any)} className="w-full rounded-lg bg-slate-800 px-3 py-2 text-sm">
          <option value="scale">scale</option>
          <option value="auto">auto</option>
        </select>
      </div>
      <div>
        <label className="block text-xs text-slate-400">OC‑SVM kernel</label>
        <select value={kernel} onChange={e => setKernel(e.target.value as any)} className="w-full rounded-lg bg-slate-800 px-3 py-2 text-sm">
          <option value="rbf">rbf</option>
          <option value="linear">linear</option>
          <option value="poly">poly</option>
          <option value="sigmoid">sigmoid</option>
        </select>
      </div>
      <div className="flex items-end">
        <button
          onClick={async () => {
            setBusy(true)
            try {
              onRun({
                perplexity,
                n_iter: nIter,
                learning_rate: 'auto',
                nu,
                gamma,
                kernel,
                random_state: 42
              })
            } finally {
              setBusy(false)
            }
          }}
          className="w-full rounded-xl bg-emerald-500/90 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-emerald-400"
        >
          {busy ? 'Running…' : 'Run t‑SNE + OC‑SVM'}
        </button>
      </div>
    </div>
  )
}
