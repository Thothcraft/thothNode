import { useCallback, useEffect, useState } from 'react'
import { get, post } from '../api'

const ts = (t?: number) =>
  t ? new Date(t * 1000).toLocaleTimeString() : '–'

interface Pred { timestamp?: number; label?: string; confidence?: number; runtime_model_id?: string }

interface Esp32Board {
  name?: string; mac?: string; role?: string
  ssid?: string; ip?: string; hostname?: string
  connected?: boolean; serial_port?: string; sensor_id?: string
  queue_depth?: number; output_dropped?: number; csi_enqueued?: number
  live?: {
    role?: string; mac?: string; ip?: string; channel?: number
    sweep?: boolean; csi_frames?: number; uptime_s?: number
    free_heap?: number; ap_rssi?: number; ping_hz?: number
    ping_sent?: number
  } | null
}

function Esp32Card() {
  const [boards, setBoards] = useState<Esp32Board[]>([])
  const [hz, setHz] = useState(200)
  const [ssid, setSsid] = useState('')
  const [psk, setPsk] = useState('')
  const [msg, setMsg] = useState('')

  const refresh = useCallback(async () => {
    const r = await get<{ boards?: Esp32Board[] }>('/api/v1/esp32')
    setBoards(r.body?.boards ?? [])
  }, [])

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  }, [refresh])

  if (!boards.length) return null

  const cmd = async (c: string) => {
    const r = await post<{ accepted?: boolean; error?: string }>(
      '/api/v1/esp32/cmd', { cmd: c })
    setMsg(r.body?.accepted === false
      ? `rejected: ${r.body?.error ?? 'unknown'}`
      : `sent ${c}`)
    setTimeout(refresh, 1500)
  }

  const setWifi = async () => {
    if (!ssid) return
    const r = await post<{ count?: number }>(
      '/api/v1/esp32/net', { ssid, psk })
    setMsg(`wifi creds → ${r.body?.count ?? 0} port(s); board rebooting`)
    setPsk('')
    setTimeout(refresh, 8000)
  }

  return (
    <div className="card">
      <h3>ESP32 radios ({boards.length})</h3>
      <table>
        <thead>
          <tr><th>board</th><th>link</th><th>stats</th></tr>
        </thead>
        <tbody>
          {boards.map((b, i) => {
            const l = b.live ?? undefined
            return (
              <tr key={b.mac ?? b.serial_port ?? i}>
                <td>
                  {b.name ?? b.mac ?? b.serial_port ?? 'esp32'}
                  <div className="muted" style={{ fontSize: 11 }}>
                    {b.role ?? 'recv'} · {b.serial_port} · {b.sensor_id}
                  </div>
                </td>
                <td>
                  {b.connected
                    ? <>{b.ssid} ({l?.ip ?? b.ip})<div className="muted"
                        style={{ fontSize: 11 }}>
                          <a href={`http://${l?.ip ?? b.ip}:5000`}
                            target="_blank" rel="noreferrer">board&nbsp;UI</a>
                        </div></>
                    : <span style={{ color: '#c66' }}>
                        {b.ssid ? `${b.ssid} — down` : 'not connected'}</span>}
                </td>
                <td className="muted" style={{ fontSize: 11 }}>
                  {l && <>
                    ping {l.ping_hz}Hz · sent {l.ping_sent} · csi {l.csi_frames}
                    · ch{l.channel} · rssi {l.ap_rssi}dBm
                    · sweep {l.sweep ? 'on' : 'off'}<br />
                    heap {Math.round((l.free_heap ?? 0) / 1024)}k
                    · up {Math.round((l.uptime_s ?? 0) / 60)}m<br />
                  </>}
                  q {b.queue_depth ?? '–'} · dropped {b.output_dropped ?? '–'}
                  · csi enq {b.csi_enqueued ?? '–'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center',
                    marginTop: 10, flexWrap: 'wrap' }}>
        <label style={{ fontSize: 12 }}>ping Hz</label>
        <input type="number" min={0} max={200} value={hz}
               style={{ width: 64 }}
               onChange={(e) => setHz(Number(e.target.value))} />
        <button onClick={() => void cmd(`ping=${hz}`)}>set rate</button>
        <button onClick={() => void cmd('sweep=1')}>sweep on</button>
        <button onClick={() => void cmd('sweep=0')}>sweep off</button>
        <span className="muted" style={{ fontSize: 11 }}>{msg}</span>
      </div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center',
                    marginTop: 6, flexWrap: 'wrap' }}>
        <input placeholder="ssid" value={ssid}
               onChange={(e) => setSsid(e.target.value)} />
        <input placeholder="psk" type="password" value={psk}
               onChange={(e) => setPsk(e.target.value)} />
        <button onClick={() => void setWifi()}>set board wifi</button>
      </div>
    </div>
  )
}

export default function StatusPage() {
  const [sensors, setSensors] = useState<any[]>([])
  const [actuators, setActuators] = useState<any[]>([])
  const [preds, setPreds] = useState<Pred[]>([])

  const refresh = useCallback(async () => {
    const [sn, ac, pr] = await Promise.all([
      get<any>('/api/sensors'),
      get<any>('/api/actuators'),
      get<any>('/api/predictions?limit=12'),
    ])
    setSensors(sn.body?.sensors ?? [])
    setActuators(ac.body?.actuators ?? [])
    setPreds(pr.body?.predictions ?? [])
  }, [])

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  }, [refresh])

  return (
    <>
      <div className="card">
        <h3>Sensors</h3>
        <table>
          <thead><tr><th>id</th><th>type</th><th>capabilities</th></tr></thead>
          <tbody>
            {sensors.map((s) => (
              <tr key={s.id}>
                <td>{s.id}</td><td>{s.type}</td>
                <td className="muted">{(s.capabilities ?? []).join(', ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>Actuators</h3>
        <table>
          <thead><tr><th>id</th><th>kind</th><th>operations</th></tr></thead>
          <tbody>
            {actuators.map((a) => (
              <tr key={a.id}>
                <td>{a.id}</td><td>{a.kind}</td>
                <td className="muted">{(a.operations ?? []).join(', ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Esp32Card />

      <div className="card">
        <h3>Recent predictions</h3>
        <table>
          <thead><tr><th>time</th><th>label</th><th>conf</th><th>model</th></tr></thead>
          <tbody>
            {preds.slice(-12).reverse().map((p, i) => (
              <tr key={i}>
                <td>{ts(p.timestamp)}</td><td>{p.label}</td>
                <td>{(p.confidence ?? 0).toFixed(2)}</td>
                <td className="muted">{p.runtime_model_id}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
