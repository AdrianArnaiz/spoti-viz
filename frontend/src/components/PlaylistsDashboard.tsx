import { useEffect, useMemo, useState } from 'react'
import Plot from 'react-plotly.js'
import { api } from '../lib/api'

export default function PlaylistsDashboard({ datasetInfo }: { datasetInfo: any | null }) {
  const [resp, setResp] = useState<any | null>(null)
  const [sim, setSim] = useState<any | null>(null)
  const [error, setError] = useState<string | null>(null)

  const run = async () => {
    setError(null)
    try {
      const [summary, similarity] = await Promise.all([
        api.post('/playlists/summary', {}),
        api.post('/playlists/similarity', {})
      ])
      setResp(summary.data)
      setSim(similarity.data)
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message)
    }
  }

  const countTrace = useMemo(() => {
    if (!resp) return null
    const names = resp.rows.map((r: any) => r.playlist)
    const counts = resp.rows.map((r: any) => r.count)
    return {
      type: 'bar',
      x: counts,
      y: names,
      orientation: 'h',
      name: 'Counts'
    }
  }, [resp])

  const heatmapTrace = useMemo(() => {
    if (!sim) return null
    return {
      type: 'heatmap',
      z: sim.similarity,
      x: sim.playlists,
      y: sim.playlists
    }
  }, [sim])

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
        <button onClick={run} className="rounded-xl bg-emerald-500/90 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-emerald-400">
          Load playlist analytics
        </button>
      </div>

      {error && <div className="rounded-xl border border-red-900 bg-red-950/40 p-3 text-sm text-red-200">{error}</div>}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-2">
          {countTrace ? (
            <Plot
              data={[countTrace]}
              layout={{
                autosize: true,
                height: 500,
                margin: { l: 120, r: 20, t: 40, b: 40 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                title: 'Top playlists by count'
              }}
              config={{ responsive: true, displaylogo: false }}
              style={{ width: '100%' }}
            />
          ) : (
            <div className="p-6 text-center text-slate-400">Click to load playlist analytics.</div>
          )}
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-2">
          {heatmapTrace ? (
            <Plot
              data={[heatmapTrace]}
              layout={{
                autosize: true,
                height: 500,
                margin: { l: 120, r: 20, t: 40, b: 40 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                title: 'Playlist similarity (cosine of mean feature vectors)'
              }}
              config={{ responsive: true, displaylogo: false }}
              style={{ width: '100%' }}
            />
          ) : (
            <div className="p-6 text-center text-slate-400">Click to load playlist analytics.</div>
          )}
        </div>
      </div>
    </div>
  )
}
