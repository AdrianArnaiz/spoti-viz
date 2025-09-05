import { useEffect, useMemo, useState } from 'react'
import Plot from 'react-plotly.js'
import { api } from '../lib/api'

export default function DistributionsDashboard({ datasetInfo }: { datasetInfo: any | null }) {
  const [feature, setFeature] = useState<string>('')
  const [groupBy, setGroupBy] = useState<'none' | 'liked' | 'playlist'>('none')
  const [bins, setBins] = useState<number>(30)
  const [kde, setKde] = useState<boolean>(true)
  const [resp, setResp] = useState<any | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // choose first numeric as default
    if (datasetInfo?.numeric_columns?.length) {
      setFeature(datasetInfo.numeric_columns[0])
    }
  }, [datasetInfo])

  const run = async () => {
    if (!feature) return
    setError(null)
    try {
      const r = await api.post('/distributions', { feature, group_by: groupBy, bins, kde })
      setResp(r.data)
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message)
    }
  }

  const traces = useMemo(() => {
    if (!resp) return []
    const traces: any[] = []
    for (const g of resp.groups) {
      const be = g.hist.bin_edges as number[]
      const counts = g.hist.counts as number[]
      const centers = be.slice(0, -1).map((b, i) => 0.5 * (be[i] + be[i + 1]))
      traces.push({
        type: 'bar',
        name: `${g.label} (n=${g.n})`,
        x: centers,
        y: counts,
        opacity: 0.6,
      })
      if (g.kde) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          name: `${g.label} KDE`,
          x: g.kde.x,
          y: g.kde.y,
        })
      }
    }
    return traces
  }, [resp])

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 rounded-xl border border-slate-800 bg-slate-900/40 p-4 md:grid-cols-4">
        <div>
          <label className="block text-xs text-slate-400">Feature</label>
          <select value={feature} onChange={e => setFeature(e.target.value)} className="w-full rounded-lg bg-slate-800 px-3 py-2 text-sm">
            {(datasetInfo?.numeric_columns || []).map((c: string) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-slate-400">Group by</label>
          <select value={groupBy} onChange={e => setGroupBy(e.target.value as any)} className="w-full rounded-lg bg-slate-800 px-3 py-2 text-sm">
            <option value="none">None</option>
            <option value="liked">Liked</option>
            <option value="playlist">Playlist (top)</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-slate-400">Bins: {bins}</label>
          <input type="range" min={10} max={100} step={5} value={bins} onChange={e => setBins(parseInt(e.target.value))} className="w-full" />
        </div>
        <div className="flex items-end">
          <button onClick={run} className="w-full rounded-xl bg-emerald-500/90 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-emerald-400">
            Run Distributions
          </button>
        </div>
      </div>

      {error && <div className="rounded-xl border border-red-900 bg-red-950/40 p-3 text-sm text-red-200">{error}</div>}

      <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-2">
        {traces.length === 0 ? (
          <div className="p-6 text-center text-slate-400">Choose a feature and run.</div>
        ) : (
          <Plot
            data={traces}
            layout={{
              autosize: true,
              height: 640,
              margin: { l: 40, r: 20, t: 40, b: 40 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              barmode: 'overlay',
              legend: { orientation: 'h', x: 0, y: 1.1 },
              title: `${resp.feature} distributions — grouped by ${resp.group_by}`
            }}
            config={{ responsive: true, displaylogo: false }}
            style={{ width: '100%' }}
          />
        )}
      </div>
    </div>
  )
}
