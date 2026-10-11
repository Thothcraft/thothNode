import { useCallback, useEffect, useRef, useState } from 'react'
import { del, get, post } from '../api'

const ts = (t?: number) =>
  t ? new Date(t * 1000).toLocaleString() : '–'

const sensorLabel = (s: any) => s.metadata?.name || s.id

/** Collect labeled baselines: pick sensors, name the state
 *  (sleep / empty_room / away …), record, then save — the node folds
 *  the capture into per-sensor descriptor stats and pushes it to
 *  Brain where context/chat agents compare live windows against it. */
export default function CalibrationsPage() {
  const [sensors, setSensors] = useState<any[]>([])
  const [checked, setChecked] = useState<Set<string>>(new Set())
  const [calibrations, setCalibrations] = useState<any[]>([])
  const [doc, setDoc] = useState<any>(null)
  const [label, setLabel] = useState('')
  const [capId, setCapId] = useState('')
  const [capStarted, setCapStarted] = useState<number | null>(null)
  const [msg, setMsg] = useState('')
  const [busy, setBusy] = useState(false)
  const [, force] = useState(0)
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const refresh = useCallback(async () => {
    const sn = await get<any>('/api/sensors')
    setSensors(sn.body?.sensors ?? [])
    const r = await get<any>('/api/calibrations')
    setCalibrations(r.body?.calibrations ?? [])
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  useEffect(() => {
    if (tickRef.current) clearInterval(tickRef.current)
    tickRef.current = capId
      ? setInterval(() => force((n) => n + 1), 1000)
      : null
    return () => { if (tickRef.current) clearInterval(tickRef.current) }
  }, [capId])

  const start = async () => {
    if (!label.trim()) { setMsg('name the label first (e.g. sleep)'); return }
    setBusy(true); setMsg('')
    const ids = [...checked]
    const r = await post<any>('/api/captures/start',
      { sensors: ids.length ? ids : null })
    setBusy(false)
    if ((r.status === 200 || r.status === 201) && r.body?.id) {
      setCapId(r.body.id)
      setCapStarted(Date.now() / 1000)
    } else setMsg(r.body?.error ?? 'capture start failed')
  }

  const stopAndSave = async () => {
    if (!capId) return
    setBusy(true); setMsg('')
    await post('/api/captures/stop', { capture_id: capId })
    const r = await post<any>('/api/calibrations',
      { label: label.trim(), capture_id: capId })
    setBusy(false)
    if (r.status === 201) {
      setMsg(`saved "${label.trim()}" — pushed to Brain`)
      setCapId(''); setCapStarted(null); setLabel('')
      refresh()
    } else setMsg(r.body?.error ?? 'calibration save failed')
  }

  const discard = async () => {
    if (!capId) return
    await post('/api/captures/stop', { capture_id: capId })
    await del(`/api/captures/${capId}`)
    setCapId(''); setCapStarted(null)
  }

  const remove = async (l: string) => {
    if (!confirm(`delete calibration "${l}"?`)) return
    await del(`/api/calibrations/${l}`)
    if (doc?.label === l) setDoc(null)
    refresh()
  }

  const view = async (l: string) => {
    const r = await get<any>(`/api/calibrations/${l}`)
    setDoc(r.status === 200 ? r.body : null)
  }

  const elapsed = capStarted ? Math.max(0, Date.now() / 1000 - capStarted) : 0
  const mm = Math.floor(elapsed / 60)
  const ss = Math.floor(elapsed % 60)

  return (
    <>
      <div className="card">
        <h3>New calibration</h3>
        <div className="muted" style={{ marginBottom: 8 }}>
          record the state under a label — saved stats become the
          baseline agents compare against
        </div>
        <div className="row">
          <input placeholder="label (sleep, empty_room, …)" size={28}
                 value={label} disabled={!!capId}
                 onChange={(e) => setLabel(e.target.value)} />
        </div>
        <div className="chips" style={{ marginTop: 8 }}>
          {sensors.map((s) => (
            <button key={s.id}
                    className={`chip ${checked.has(s.id) ? 'on' : ''}`}
                    disabled={!!capId}
                    onClick={() => {
                      const next = new Set(checked)
                      if (next.has(s.id)) next.delete(s.id)
                      else next.add(s.id)
                      setChecked(next)
                    }}>
              {sensorLabel(s)}{' '}
              <span className="muted">{s.type}</span>
            </button>
          ))}
          {!sensors.length && <span className="muted">no sensors</span>}
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          {!capId ? (
            <>
              <button className="go" disabled={busy} onClick={start}>
                ● record baseline
              </button>
              <span className="muted">
                {checked.size
                  ? `${checked.size} sensor${checked.size > 1 ? 's' : ''} selected`
                  : 'no selection = all sensors'}
              </span>
            </>
          ) : (
            <>
              <span className="pill live">● recording {mm}m {ss}s</span>
              <button className="go" disabled={busy} onClick={stopAndSave}>
                stop & save "{label}"
              </button>
              <button className="danger" onClick={discard}>discard</button>
            </>
          )}
        </div>
      </div>

      <div className="card">
        <h3>Saved calibrations</h3>
        {!calibrations.length && (
          <span className="muted">none yet — record a labeled baseline above.</span>
        )}
        {calibrations.length > 0 && (
          <table>
            <thead>
              <tr><th>label</th><th>duration</th><th>sensors</th>
                  <th>created</th><th /></tr>
            </thead>
            <tbody>
              {calibrations.map((c) => (
                <tr key={c.label} className={doc?.label === c.label ? 'sel' : ''}>
                  <td><a href="#" onClick={(e) => { e.preventDefault(); void view(c.label) }}>
                    {c.label}</a></td>
                  <td>{c.duration_s ? `${Math.round(c.duration_s / 60)}m` : '–'}</td>
                  <td><small>{(c.sensors ?? []).join(', ') || '–'}</small></td>
                  <td>{ts(c.created_at)}</td>
                  <td className="row-cells">
                    <button className="mini danger"
                            onClick={() => remove(c.label)}>delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {doc && (
        <div className="card">
          <h3>{doc.label} — descriptor baseline</h3>
          <span className="stat">capture <b>{doc.capture_id || '–'}</b></span>
          <span className="stat">duration <b>{doc.duration_s}s</b></span>
          <span className="stat">sensors <b>{Object.keys(doc.sensors ?? {}).length}</b></span>
          <pre style={{ maxHeight: 360, marginTop: 10 }}>
            {JSON.stringify(doc.sensors ?? {}, null, 2)}
          </pre>
        </div>
      )}
      {msg && <div className="muted" style={{ marginTop: 6 }}>{msg}</div>}
    </>
  )
}
