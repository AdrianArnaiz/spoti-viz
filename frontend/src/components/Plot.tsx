\
import Plot from 'react-plotly.js'

type Point = {
  id: number
  x: number
  y: number
  score: number
  inlier: boolean
  [k: string]: any
}

export default function ScatterPlot({
  data,
  colorBy,
  metaColumns
}: {
  data: Point[],
  colorBy: 'inlier' | 'playlist' | 'none',
  metaColumns: string[]
}) {
  // Split into traces by category
  let traces: any[] = []
  if (colorBy === 'inlier') {
    const groups = { inlier: data.filter(d => d.inlier), outlier: data.filter(d => !d.inlier) }
    traces = Object.entries(groups).map(([label, pts]) => ({
      type: 'scattergl',
      mode: 'markers',
      name: label,
      x: pts.map(p => p.x),
      y: pts.map(p => p.y),
      text: pts.map(p => hoverText(p, metaColumns)),
      hoverinfo: 'text',
      marker: { size: 7, opacity: 0.85 },
    }))
  } else if (colorBy === 'playlist') {
    const byPlaylist = new Map<string, Point[]>()
    data.forEach(p => {
      const key = (p.playlist ?? 'Unknown').toString()
      if (!byPlaylist.has(key)) byPlaylist.set(key, [])
      byPlaylist.get(key)!.push(p)
    })
    traces = Array.from(byPlaylist.entries()).map(([label, pts]) => ({
      type: 'scattergl',
      mode: 'markers',
      name: label,
      x: pts.map(p => p.x),
      y: pts.map(p => p.y),
      text: pts.map(p => hoverText(p, metaColumns)),
      hoverinfo: 'text',
      marker: { size: 7, opacity: 0.85 },
    }))
  } else {
    traces = [{
      type: 'scattergl',
      mode: 'markers',
      name: 'points',
      x: data.map(p => p.x),
      y: data.map(p => p.y),
      text: data.map(p => hoverText(p, metaColumns)),
      hoverinfo: 'text',
      marker: { size: 7, opacity: 0.85 },
    }]
  }

  return (
    <Plot
      data={traces}
      layout={{
        autosize: true,
        height: 640,
        margin: { l: 40, r: 20, t: 40, b: 40 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { showgrid: false, zeroline: false },
        yaxis: { showgrid: false, zeroline: false },
        legend: { orientation: 'h', x: 0, y: 1.1 },
        hovermode: 'closest',
        title: 't-SNE projection • hover for details'
      }}
      config={{
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d'],
      }}
      style={{ width: '100%' }}
    />
  )
}

function hoverText(p: any, metaCols: string[]) {
  const name = p.track_name || p.song_name || p.name || p.track || 'Track'
  const artist = p.artist || p.artists || ''
  const playlist = p.playlist ? `\nPlaylist: ${p.playlist}` : ''
  const inlier = p.inlier ? 'inlier' : 'outlier'
  return `${name} — ${artist}\nSVM: ${inlier} (score ${p.score.toFixed(3)})${playlist}`
}
