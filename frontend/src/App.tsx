\
import { Tab } from '@headlessui/react'
import { motion } from 'framer-motion'
import { UploadIcon } from '@heroicons/react/24/outline'
import TsneOcsvmDashboard from './components/TsneOcsvmDashboard'
import FileUploader from './components/FileUploader'
import { useState } from 'react'

function classNames(...classes: (string | boolean | undefined)[]) {
  return classes.filter(Boolean).join(' ')
}

export default function App() {
  const [datasetInfo, setDatasetInfo] = useState<any | null>(null)

  return (
    <div className="min-h-screen text-slate-100">
      <header className="sticky top-0 z-30 border-b border-slate-800 bg-slate-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-2xl bg-emerald-500/20 ring-1 ring-emerald-500/40">
              <svg viewBox="0 0 128 128" className="h-6 w-6"><circle cx="64" cy="64" r="60" fill="#10b981"/><path d="M36 64c20 12 36-12 56 0" stroke="white" strokeWidth="8" fill="none"/></svg>
            </div>
            <div>
              <h1 className="text-lg font-semibold">Spotify Taste Analysis</h1>
              <p className="text-xs text-slate-400">t‑SNE + One‑Class SVM • Interactive Dashboards</p>
            </div>
          </div>
          <a href="https://plotly.com" target="_blank" rel="noreferrer" className="text-xs text-slate-400 hover:text-slate-200">Powered by Plotly</a>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 pb-24 pt-6">
        <motion.section
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, ease: 'easeOut' }}
          className="mb-6 rounded-2xl border border-slate-800 bg-slate-900/40 p-4"
        >
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-base font-medium">Upload your Spotify CSV</h2>
              <p className="text-sm text-slate-400">We detect numeric features automatically; you can tune parameters later.</p>
            </div>
            <FileUploader onUploaded={setDatasetInfo} />
          </div>

          {datasetInfo && (
            <div className="mt-4 overflow-x-auto rounded-xl border border-slate-800">
              <table className="min-w-full text-sm">
                <thead className="bg-slate-900/60">
                  <tr className="text-left">
                    <th className="px-3 py-2 font-medium">Rows</th>
                    <th className="px-3 py-2 font-medium">Cols</th>
                    <th className="px-3 py-2 font-medium">Numeric</th>
                    <th className="px-3 py-2 font-medium">Label candidates</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="px-3 py-2">{datasetInfo.rows}</td>
                    <td className="px-3 py-2">{datasetInfo.cols}</td>
                    <td className="px-3 py-2">{datasetInfo.numeric_columns?.length ?? 0}</td>
                    <td className="px-3 py-2">{datasetInfo.label_candidates?.join(', ') || '—'}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </motion.section>

        <Tab.Group>
          <Tab.List className="mb-4 flex gap-2">
            {['t‑SNE • One‑Class SVM', 'Distributions (soon)', 'Playlists (soon)'].map((label, idx) => (
              <Tab key={idx} className={({ selected }) =>
                classNames(
                  'rounded-xl px-4 py-2 text-sm outline-none ring-1 ring-slate-800',
                  selected ? 'bg-emerald-500/20 text-emerald-200 ring-emerald-500/40' : 'bg-slate-900/40 text-slate-300 hover:bg-slate-900/60'
                )
              }>
                {label}
              </Tab>
            ))}
          </Tab.List>

          <Tab.Panels>
            <Tab.Panel>
              <TsneOcsvmDashboard datasetInfo={datasetInfo} />
            </Tab.Panel>
            <Tab.Panel>
              <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 text-slate-300">
                <p>Coming soon: interactive histograms, KDEs, feature distributions by playlist &amp; liked status.</p>
              </div>
            </Tab.Panel>
            <Tab.Panel>
              <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 text-slate-300">
                <p>Coming soon: playlist comparisons, similarity heatmaps, and recommendation spots.</p>
              </div>
            </Tab.Panel>
          </Tab.Panels>
        </Tab.Group>
      </main>
    </div>
  )
}
