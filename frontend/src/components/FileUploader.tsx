\
import { useRef, useState } from 'react'
import { api } from '../lib/api'

export default function FileUploader({ onUploaded }: { onUploaded: (info: any) => void }) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const onSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await api.post('/upload', form, { headers: { 'Content-Type': 'multipart/form-data' }})
      onUploaded(res.data)
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message)
    } finally {
      setBusy(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  return (
    <div className="flex items-center gap-3">
      <label className="inline-flex cursor-pointer items-center gap-3 rounded-xl border border-slate-700 bg-slate-900/60 px-4 py-2 text-sm hover:bg-slate-900">
        <input ref={inputRef} type="file" accept=".csv" className="hidden" onChange={onSelect} />
        <span>Choose CSV…</span>
      </label>
      {busy && <span className="text-xs text-slate-400">Uploading…</span>}
      {error && <span className="text-xs text-red-300">{error}</span>}
    </div>
  )
}
