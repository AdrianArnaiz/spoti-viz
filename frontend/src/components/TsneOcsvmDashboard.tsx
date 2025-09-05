\
import { useEffect, useMemo, useState } from 'react'
import ControlsPanel, { type Params } from './ControlsPanel'
import ScatterPlot from './Plot'
import { api } from '../lib/api'

export default function TsneOcsvmDashboard({ datasetInfo }: { datasetInfo: any | null }) {
  const [points, setPoints] = useState<any[]>([])
  const [metaCols, setMetaCols] = useState<string[]>([])
  const [colorBy, setColorBy] = useState<'inlier' | 'playlist' | 'none'>('inlier')
  const [info, setInfo] = useState<any | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Reset when dataset changes
    setPoints([])
    setInfo(null)
    setError(null)
  }, [datasetInfo])

  const run = async (p: Params) => {
    setError(null)
    try {
      const res = await api.post('/tsne-ocsvm', p)
      setPoints(res.data.points)
      setMetaCols(res.data.meta_columns)
      setInfo({
        n_samples: res.data.n_samples,
        perplexity_used: res.data.perplexity_used,
        ocsvm_params: res.data.ocsvm_params
      })
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message)
    }
  }

  return (
    <div className="space-y-4">
      <ControlsPanel onRun={run} />

      {error && <div className="rounded-xl border border-red-900 bg-red-950/40 p-3 text-sm text-red-200">{error}</div>}

      {info && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3 text-sm text-slate-300">
          <div>Samples: {info.n_samples} • Perplexity used: {info.perplexity_used.toFixed(1)} • OC‑SVM: ν={info.ocsvm_params.nu}, γ={String(info.ocsvm_params.gamma)}, kernel={info.ocsvm_params.kernel}</div>
        </div>
      )}

      <div className="flex items-center gap-3">
        <label className="text-sm text-slate-300">Color by:</label>
        <select className="rounded-lg bg-slate-800 px-3 py-2 text-sm" value={colorBy} onChange={(e) => setColorBy(e.target.value as any)}>
          <option value="inlier">OC‑SVM Inlier/Outlier</option>
          <option value="playlist">Playlist</option>
          <option value="none">None</option>
        </select>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-2">
        {points.length === 0 ? (
          <div className="p-6 text-center text-slate-400">Run the analysis to see the projection.</div>
        ) : (
          <ScatterPlot data={points} colorBy={colorBy} metaColumns={metaCols} />
        )}
      </div>
    </div>
  )
}
