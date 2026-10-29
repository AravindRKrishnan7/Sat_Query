import { useEffect, useMemo, useState } from "react"

type View = "welcome" | "infrastructure" | "vegetation"
type JobStatus = "queued" | "running" | "success" | "partial" | "failed"

type Job = {
  analysis_id: string
  workflow: string
  status: JobStatus
  progress: number
  message: string
  error?: string | null
  result?: any
}

type UploadObservation = {
  id: string
  file: File | null
  date: string
  preview: string | null
}

type ConfirmedArea = {
  label: string
  bounds_wgs84: number[]
  area_square_km: number
}

type PresetContext = {
  area: ConfirmedArea
  boundary_source: string
  preview: null | {
    year: number
    acquisition_date: string
    image_url: string
    source_record_url: string
    label: string
  }
  preview_status: "matching_saved_observation" | "unavailable"
  preview_unavailable_reason: string | null
  analysis_performed: false
  retrieval_performed: false
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8010").replace(/\/$/, "")

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`)
  return payload as T
}

function asset(path?: string | null) {
  return path ? `${API_BASE}${path}` : ""
}

function newObservation(): UploadObservation {
  return { id: crypto.randomUUID(), file: null, date: "", preview: null }
}

function worldPixel(longitude: number, latitude: number, zoom: number) {
  const scale = 256 * 2 ** zoom
  const clamped = Math.max(-85.05112878, Math.min(85.05112878, latitude))
  const radians = clamped * Math.PI / 180
  return {
    x: (longitude + 180) / 360 * scale,
    y: (1 - Math.log(Math.tan(radians) + 1 / Math.cos(radians)) / Math.PI) / 2 * scale,
  }
}

function GeographicMap({ area }: { area: ConfirmedArea }) {
  const [tilesFailed, setTilesFailed] = useState(false)
  const zoom = 14
  const [west, south, east, north] = area.bounds_wgs84
  const center = worldPixel((west + east) / 2, (south + north) / 2, zoom)
  const northWest = worldPixel(west, north, zoom)
  const southEast = worldPixel(east, south, zoom)
  const centerTileX = Math.floor(center.x / 256)
  const centerTileY = Math.floor(center.y / 256)
  const tiles = useMemo(() => {
    const values: { x: number; y: number; left: number; top: number }[] = []
    for (let y = centerTileY - 2; y <= centerTileY + 2; y += 1) {
      for (let x = centerTileX - 2; x <= centerTileX + 2; x += 1) {
        values.push({ x, y, left: x * 256 - center.x, top: y * 256 - center.y })
      }
    }
    return values
  }, [center.x, center.y, centerTileX, centerTileY])
  useEffect(() => setTilesFailed(false), [west, south, east, north])
  return <figure className={`geographic-map ${tilesFailed ? "tiles-failed" : ""}`} aria-label={`Map of ${area.label} with confirmed rectangle`}>
    <div className="map-tile-layer" aria-hidden="true">
      {tiles.map((tile) => <img
        key={`${tile.x}-${tile.y}`}
        src={`https://tile.openstreetmap.org/${zoom}/${tile.x}/${tile.y}.png`}
        alt=""
        loading="lazy"
        onError={() => setTilesFailed(true)}
        style={{ left: `calc(50% + ${tile.left}px)`, top: `calc(50% + ${tile.top}px)` }}
      />)}
    </div>
    <div
      className="confirmed-rectangle"
      aria-hidden="true"
      style={{
        left: `calc(50% + ${northWest.x - center.x}px)`,
        top: `calc(50% + ${northWest.y - center.y}px)`,
        width: `${southEast.x - northWest.x}px`,
        height: `${southEast.y - northWest.y}px`,
      }}
    ><span>{area.label}</span></div>
    {tilesFailed && <div className="map-tile-error"><Icon name="info" /><span>Basemap tiles are unavailable. The confirmed rectangle and exact bounds remain shown.</span></div>}
    <figcaption>
      <span>Basemap: © OpenStreetMap contributors · geographic context only</span>
      <strong>W/S/E/N: {area.bounds_wgs84.join(", ")}</strong>
    </figcaption>
  </figure>
}

function Icon({ name }: { name: "spark" | "building" | "leaf" | "plus" | "check" | "report" | "send" | "info" | "download" }) {
  const paths = {
    spark: <path d="m12 2 1.2 4.4a6 6 0 0 0 4.4 4.4L22 12l-4.4 1.2a6 6 0 0 0-4.4 4.4L12 22l-1.2-4.4a6 6 0 0 0-4.4-4.4L2 12l4.4-1.2a6 6 0 0 0 4.4-4.4L12 2Z" />,
    building: <path d="M4 21V8l8-4 8 4v13M8 11h2m4 0h2m-8 4h2m4 0h2M9 21v-3h6v3" />,
    leaf: <path d="M20 4C12 4 5 8 5 15c0 2 1 4 3 5 5-2 8-6 10-11-1 6-4 10-10 11M5 15c-1-1-2-3-1-6" />,
    plus: <path d="M12 5v14M5 12h14" />,
    check: <path d="m5 12 4 4L19 6" />,
    report: <path d="M6 2h9l4 4v16H6zM14 2v5h5M9 12h7m-7 4h7" />,
    send: <path d="m3 3 18 9-18 9 3-9-3-9Zm3 9h15" />,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6m0-10v.01" /></>,
    download: <path d="M12 3v12m0 0 4-4m-4 4-4-4M5 21h14" />,
  }
  return <svg className="icon" width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">{paths[name]}</svg>
}

function Brand() {
  return <div className="brand"><span className="brand-mark"><span /></span><span>GeoLens</span></div>
}

function Sidebar({ view, onNavigate }: { view: View; onNavigate: (view: View) => void }) {
  return <aside className="sidebar">
    <Brand />
    <button className="new-analysis" onClick={() => onNavigate("welcome")}><Icon name="plus" /> New analysis</button>
    <nav aria-label="Primary navigation">
      <p className="nav-label">Workspaces</p>
      <button className={`nav-item ${view === "infrastructure" ? "active" : ""}`} onClick={() => onNavigate("infrastructure")}><Icon name="building" /><span>Buildings and infrastructure</span></button>
      <button className={`nav-item ${view === "vegetation" ? "active" : ""}`} onClick={() => onNavigate("vegetation")}><Icon name="leaf" /><span>Vegetation history</span></button>
    </nav>
    <div className="sidebar-footer live"><span className="sample-pulse" /><span><strong>Local application</strong><small>Real saved analysis</small></span></div>
  </aside>
}

function Header({ title, subtitle }: { title: string; subtitle: string }) {
  return <header className="workspace-header"><div><h1>{title}</h1><p>{subtitle}</p></div><span className="live-badge"><span /> Local services</span></header>
}

function ErrorNotice({ message }: { message: string | null }) {
  if (!message) return null
  return <div className="error-notice" role="alert"><Icon name="info" /><span>{message}</span></div>
}

function Progress({ job }: { job: Job }) {
  return <div className="assistant-card run-card">
    <span className="card-kicker">Analysis {job.analysis_id}</span>
    <h2>{job.message}</h2>
    <div className="progress-track"><span style={{ width: `${job.progress}%` }} /></div>
    <div className="progress-meta"><span>{job.status}</span><strong>{job.progress}%</strong></div>
  </div>
}

function DownloadLink({ href, children }: { href?: string | null; children: React.ReactNode }) {
  if (!href) return null
  return <a className="download-link" href={asset(href)} download><Icon name="download" />{children}</a>
}

function FollowUps({ analysisId, comparisonId }: { analysisId: string; comparisonId?: string }) {
  const [question, setQuestion] = useState("")
  const [messages, setMessages] = useState<{ question: string; answer: string; fallback?: string | null }[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  async function submit(event: React.FormEvent) {
    event.preventDefault()
    if (!question.trim() || busy) return
    const submitted = question.trim()
    setBusy(true); setError(null)
    try {
      const result = await api<any>(`/api/analyses/${analysisId}/followups`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: submitted, selected_comparison_id: comparisonId || null }),
      })
      setMessages((current) => [...current, { question: submitted, answer: result.answer, fallback: result.fallback_reason }])
      setQuestion("")
    } catch (caught) { setError((caught as Error).message) } finally { setBusy(false) }
  }
  return <section className="follow-up-section">
    <h3>Ask about this completed analysis</h3>
    {messages.map((message, index) => <div className="follow-up-thread" key={index}><div className="user-message">{message.question}</div><div className="follow-up-answer"><Icon name="spark" /><div><p>{message.answer}</p>{message.fallback && <small>Deterministic fallback: {message.fallback}</small>}</div></div></div>)}
    <form className="composer inline" onSubmit={submit}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about the saved evidence…" maxLength={2000} /><button aria-label="Send follow-up" disabled={busy}><Icon name="send" /></button></form>
    <ErrorNotice message={error} />
  </section>
}

function ReportAction({ analysisId }: { analysisId: string }) {
  const [url, setUrl] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  async function build() {
    setBusy(true); setError(null)
    try { const result = await api<any>(`/api/analyses/${analysisId}/report`, { method: "POST" }); setUrl(result.report_url) }
    catch (caught) { setError((caught as Error).message) } finally { setBusy(false) }
  }
  return <div className="report-action"><button className="secondary-button" onClick={build} disabled={busy}><Icon name="report" />{busy ? "Building report…" : "Build illustrated PDF"}</button><DownloadLink href={url}>Download PDF report</DownloadLink><ErrorNotice message={error} /></div>
}

function useJob(analysisId: string | null, job: Job | null, setJob: (job: Job) => void) {
  useEffect(() => {
    if (!analysisId || !job || !["queued", "running"].includes(job.status)) return
    const timer = window.setInterval(async () => {
      try { setJob(await api<Job>(`/api/jobs/${analysisId}`)) } catch { /* retain last state */ }
    }, 900)
    return () => window.clearInterval(timer)
  }, [analysisId, job?.status, setJob])
}

function Infrastructure() {
  const [question, setQuestion] = useState("What building and road changes does the model predict?")
  const [observations, setObservations] = useState<UploadObservation[]>([newObservation(), newObservation()])
  const [resize, setResize] = useState(false)
  const [endpoint, setEndpoint] = useState(false)
  const [aligned, setAligned] = useState(false)
  const [plan, setPlan] = useState<any>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedComparison, setSelectedComparison] = useState("")
  const invalidate = () => { setPlan(null); setJob(null); setSelectedComparison(""); setError(null) }
  const changeQuestion = (value: string) => { invalidate(); setQuestion(value) }
  function updateObservation(id: string, change: Partial<UploadObservation>) { invalidate(); setObservations((items) => items.map((item) => item.id === id ? { ...item, ...change } : item)) }
  function selectFile(id: string, file: File | null) {
    const current = observations.find((item) => item.id === id)
    if (current?.preview) URL.revokeObjectURL(current.preview)
    updateObservation(id, { file, preview: file ? URL.createObjectURL(file) : null })
  }
  async function validate() {
    setBusy(true); setError(null)
    try {
      const form = new FormData()
      form.set("question", question)
      form.set("dates", JSON.stringify(observations.map((item) => item.date)))
      form.set("resize", String(resize)); form.set("include_endpoint", String(endpoint)); form.set("alignment_confirmed", String(aligned))
      observations.forEach((item) => { if (item.file) form.append("files", item.file) })
      setPlan(await api<any>("/api/infrastructure/plans", { method: "POST", body: form }))
    } catch (caught) { setError((caught as Error).message) } finally { setBusy(false) }
  }
  async function run() {
    if (!plan) return
    setBusy(true); setError(null)
    try { setJob(await api<Job>(`/api/plans/${plan.analysis_id}/run`, { method: "POST" })) }
    catch (caught) { setError((caught as Error).message) } finally { setBusy(false) }
  }
  useJob(plan?.analysis_id || null, job, setJob)
  const result = job?.result
  const comparisons = result?.comparisons || []
  const selected = comparisons.find((item: any) => item.id === selectedComparison) || comparisons[0]
  const before = result?.observations?.find((item: any) => item.id === selected?.before_observation)
  const after = result?.observations?.find((item: any) => item.id === selected?.after_observation)
  useEffect(() => { if (comparisons.length && !selectedComparison) setSelectedComparison(comparisons[0].id) }, [comparisons.length, selectedComparison])

  return <main className="workspace"><Header title="Buildings and infrastructure" subtitle="Real MCI comparisons for 2–4 aligned RGB observations" />
    <div className="workspace-grid">
      <section className="chat-panel form-panel"><div className="chat-scroll">
        <span className="card-kicker">1 · Request and uploads</span>
        <label className="field"><span>What would you like to know?</span><textarea value={question} onChange={(event) => changeQuestion(event.target.value)} maxLength={2000} /></label>
        <p className="helper">Every observation needs a user-supplied date. GeoLens sorts the sequence before execution.</p>
        {observations.map((item, index) => <div className="upload-card" key={item.id}>
          <div><strong>Observation {index + 1}</strong>{observations.length > 2 && <button className="text-button" onClick={() => { invalidate(); setObservations((items) => items.filter((value) => value.id !== item.id)) }}>Remove</button>}</div>
          <input aria-label={`Image ${index + 1}`} type="file" accept="image/png,image/jpeg" onChange={(event) => selectFile(item.id, event.target.files?.[0] || null)} />
          <input aria-label={`Acquisition date ${index + 1}`} type="date" value={item.date} onChange={(event) => updateObservation(item.id, { date: event.target.value })} />
          {item.file && <small>{item.file.name} · {(item.file.size / 1024 / 1024).toFixed(2)} MB</small>}
        </div>)}
        {observations.length < 4 && <button className="secondary-button full" onClick={() => { invalidate(); setObservations((items) => [...items, newObservation()]) }}><Icon name="plus" />Add observation</button>}
        <label className="check-control"><input type="checkbox" checked={resize} onChange={(event) => { invalidate(); setResize(event.target.checked) }} /><span><strong>Prepare 256×256 copies when needed</strong><small>Resizing does not align images or restore spatial detail.</small></span></label>
        {observations.length > 2 && <label className="check-control"><input type="checkbox" checked={endpoint} onChange={(event) => { invalidate(); setEndpoint(event.target.checked) }} /><span><strong>Also compare first to last</strong><small>This is an independent endpoint prediction, not a sum.</small></span></label>}
        <label className="check-control"><input type="checkbox" checked={aligned} onChange={(event) => { invalidate(); setAligned(event.target.checked) }} /><span><strong>I confirm every image shows the same extent and is already aligned.</strong></span></label>
        <button className="primary-button full" disabled={busy || observations.some((item) => !item.file || !item.date) || !aligned || !question.trim()} onClick={validate}>{busy ? "Validating…" : "Validate and review"}</button>
        <ErrorNotice message={error} />
        {plan && !job && <div className="assistant-card"><span className="card-kicker">2 · Validated plan</span><h2>{plan.comparison_count} independent comparison{plan.comparison_count === 1 ? "" : "s"}</h2><p>Chronological order and input hashes are recorded. Review the prepared copies, then run explicitly.</p><button className="primary-button full" onClick={run} disabled={busy}>Run timeline analysis</button></div>}
        {job && ["queued", "running"].includes(job.status) && <Progress job={job} />}
        {job?.status === "failed" && <ErrorNotice message={job.error || "Analysis failed"} />}
        {result && <><div className="answer-card"><div className="answer-heading"><Icon name="spark" /><span><strong>GeoLens answer</strong><small>Saved model result</small></span></div><div className="answer-text markdown-answer">{result.answer}</div><div className="caution"><Icon name="info" /><span>{result.limitations[0]}</span></div></div><FollowUps analysisId={result.analysis_id} comparisonId={selected?.id} /><details className="technical"><summary>View evidence and technical details</summary><p>The raw captions, masks, pixel populations, hashes and timings remain in the child and parent records.</p><div className="download-row"><DownloadLink href={result.answer_url}>Timeline answer</DownloadLink><DownloadLink href={result.record_url}>Execution record</DownloadLink></div></details><ReportAction analysisId={result.analysis_id} /></>}
      </div></section>
      <section className="evidence-panel"><div className="panel-heading"><div><span className="panel-kicker">Evidence</span><h2>{result ? "Saved comparisons" : plan ? "Chronological review" : "Upload preview"}</h2></div>{job && <span className={`status-badge ${job.status}`}>{job.status}</span>}</div>
        {!plan && <div className="preview-grid">{observations.map((item, index) => <div className="preview-item" key={item.id}>{item.preview ? <img src={item.preview} alt={`Uploaded observation ${index + 1}`} /> : <div className="empty-preview">Observation {index + 1}</div>}<strong>{item.date || "Date required"}</strong></div>)}</div>}
        {plan && !result && <div className="image-timeline">{plan.chronological_observations.map((item: any) => <div className="timeline-item" key={item.prepared_sha256}><img src={asset(item.preview_url)} alt={`Prepared ${item.acquisition_date}`} /><div className="timeline-meta"><span>{item.chronological_order}</span><div><strong>{item.acquisition_date}</strong><small>{item.filename}{item.resized ? " · resized" : " · unchanged"}</small></div></div></div>)}</div>}
        {result && <><div className="interval-tabs">{comparisons.map((item: any) => <button className={selected?.id === item.id ? "active" : ""} onClick={() => setSelectedComparison(item.id)} key={item.id}>{item.before_date} → {item.after_date}{item.kind === "endpoint" ? " · endpoint" : ""}</button>)}</div>{selected?.status === "success" ? <><div className="side-view real-images"><figure><img src={asset(before?.image_url)} alt={`Before ${selected.before_date}`} /><figcaption>Before · {selected.before_date}</figcaption></figure><figure><img src={asset(after?.image_url)} alt={`After ${selected.after_date}`} /><figcaption>After · {selected.after_date}</figcaption></figure></div><figure className="overlay-figure"><img src={asset(selected.overlay_url)} alt="Model prediction overlay" /><figcaption>Red: predicted building changes · Yellow: predicted road changes</figcaption></figure><div className="interval-answer">{selected.answer}</div><div className="download-row"><DownloadLink href={selected.overlay_url}>Overlay</DownloadLink><DownloadLink href={selected.caption_url}>Raw caption</DownloadLink><DownloadLink href={selected.execution_record_url}>Child record</DownloadLink></div></> : <ErrorNotice message={selected?.error || "This interval did not complete."} />}</>}
      </section>
    </div>
  </main>
}

function Vegetation() {
  const [question, setQuestion] = useState("Compare vegetation from 2021 to 2025.")
  const [custom, setCustom] = useState(false)
  const [bounds, setBounds] = useState<number[]>([])
  const [preset, setPreset] = useState<PresetContext | null>(null)
  const [presetError, setPresetError] = useState<string | null>(null)
  const [plan, setPlan] = useState<any>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedYear, setSelectedYear] = useState<number | null>(null)
  const [selectedDifference, setSelectedDifference] = useState("")
  const invalidate = () => { setPlan(null); setJob(null); setError(null) }
  useEffect(() => {
    let active = true
    api<PresetContext>("/api/vegetation/preset")
      .then((context) => {
        if (!active) return
        setPreset(context)
        setBounds((current) => current.length === 4 ? current : context.area.bounds_wgs84)
        setPresetError(null)
      })
      .catch((caught) => {
        if (active) setPresetError((caught as Error).message)
      })
    return () => { active = false }
  }, [])
  async function validate() {
    setBusy(true); setError(null)
    try {
      setPlan(await api<any>("/api/vegetation/plans", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, bounds: custom ? bounds : null, area_label: custom ? "Custom confirmed area" : null }) }))
    } catch (caught) { setError((caught as Error).message) } finally { setBusy(false) }
  }
  async function run() {
    setBusy(true); setError(null)
    try { setJob(await api<Job>(`/api/plans/${plan.analysis_id}/run`, { method: "POST" })) }
    catch (caught) { setError((caught as Error).message) } finally { setBusy(false) }
  }
  useJob(plan?.analysis_id || null, job, setJob)
  const result = job?.result
  const available = result?.observations?.filter((item: any) => item.status === "available") || []
  const observation = available.find((item: any) => item.year === selectedYear) || available[0]
  const differences = result?.differences || []
  const difference = differences.find((item: any) => item.label === selectedDifference) || differences[0]
  const confirmedArea: ConfirmedArea | null = result
    ? { label: result.area_label, ...result.area }
    : plan?.area || (!custom ? preset?.area : null) || null
  const mapArea: ConfirmedArea | null = confirmedArea
  useEffect(() => { if (available.length && selectedYear === null) setSelectedYear(available[0].year) }, [available.length, selectedYear])
  useEffect(() => { if (differences.length && !selectedDifference) setSelectedDifference(differences[0].label) }, [differences.length, selectedDifference])
  return <main className="workspace"><Header title="Vegetation history" subtitle="Validated historical Sentinel-2 greenness analysis" />
    <div className="workspace-grid vegetation-grid">
      <section className="chat-panel form-panel"><div className="chat-scroll">
        <span className="card-kicker">1 · Confirmed area</span>
        {presetError && <ErrorNotice message={`The validated preset could not be loaded: ${presetError}`} />}
        {confirmedArea ? <div className="area-card"><div><strong>{confirmedArea.label}</strong><small>WGS84 [{confirmedArea.bounds_wgs84.join(", ")}] · west, south, east, north</small></div><span>≈ {confirmedArea.area_square_km.toFixed(3)} km²</span></div> : <div className="area-card loading"><div><strong>{custom ? "Custom bounds draft" : "Loading validated preset…"}</strong><small>{custom && bounds.length === 4 ? `WGS84 [${bounds.join(", ")}]` : "Waiting for the local API"}</small></div></div>}
        <p className="boundary-confirmation">{plan ? "Confirmed by the backend. This exact boundary is recorded in the reviewed plan." : custom ? "This is a draft. Validate the request before it can be used by a plan." : preset ? "Confirmed by the backend preset. The next plan will use this exact rectangle." : "No boundary will be used until the backend confirms it."}</p>
        <details className="technical"><summary>Use a custom rectangle</summary><p>Enter west, south, east and north. The same server-side area validator and 25 km² limit apply.</p><label className="check-control"><input type="checkbox" checked={custom} onChange={(event) => { invalidate(); setCustom(event.target.checked); if (!event.target.checked && preset) setBounds(preset.area.bounds_wgs84) }} /><span><strong>Use custom bounds</strong></span></label>{custom && <div className="bounds-grid">{["West", "South", "East", "North"].map((label, index) => <label key={label}><span>{label}</span><input type="number" step="0.0001" value={bounds[index] ?? ""} onChange={(event) => { invalidate(); setBounds((current) => [0, 1, 2, 3].map((position) => position === index ? Number(event.target.value) : current[position])) }} /></label>)}</div>}</details>
        <span className="card-kicker">2 · Ask and review</span>
        <label className="field"><span>What would you like to compare?</span><textarea value={question} onChange={(event) => { invalidate(); setQuestion(event.target.value) }} maxLength={2000} /></label>
        <button className="primary-button full" onClick={validate} disabled={busy || !question.trim() || (!custom && !preset) || bounds.length !== 4}>{busy ? "Validating…" : "Validate request"}</button>
        <ErrorNotice message={error} />
        {plan && !job && <div className="assistant-card"><span className="card-kicker">Validated plan · {plan.status}</span><p>{plan.summary}</p>{plan.status === "validated" ? <button className="primary-button full" onClick={run} disabled={busy}>Run historical analysis</button> : <div className="caution"><Icon name="info" /><span>This plan cannot run until the displayed request issue is resolved.</span></div>}<details className="technical"><summary>Plan evidence</summary><pre>{JSON.stringify(plan.plan, null, 2)}</pre></details></div>}
        {job && ["queued", "running"].includes(job.status) && <Progress job={job} />}
        {job?.status === "failed" && <ErrorNotice message={job.error || "Analysis failed"} />}
        {result && <><div className="answer-card vegetation-answer"><div className="answer-heading"><Icon name="spark" /><span><strong>GeoLens answer</strong><small>{result.answer_validation.fallback_used ? "Checked deterministic fallback" : "Validated constrained claims"}</small></span></div><p className="answer-text">{result.answer}</p><div className="caution"><Icon name="info" /><span>NDVI is related to vegetation greenness; it does not establish tree counts, vegetation-covered area, plant health or cause.</span></div></div><FollowUps analysisId={result.analysis_id} comparisonId={difference?.label} /><details className="technical"><summary>View evidence and technical details</summary><p>Available years: {result.available_years.join(", ") || "none"}. Missing years: {result.missing_years.join(", ") || "none"}.</p><div className="download-row"><DownloadLink href={result.answer_url}>Answer</DownloadLink><DownloadLink href={result.record_url}>Analysis record</DownloadLink><DownloadLink href={result.assistant_record_url}>Assistant record</DownloadLink></div></details><ReportAction analysisId={result.analysis_id} /></>}
      </div></section>
      <section className="evidence-panel vegetation-map-panel"><div className="panel-heading"><div><span className="panel-kicker">Area and evidence</span><h2>{mapArea?.label || "Waiting for confirmed area"}</h2></div>{job && <span className={`status-badge ${job.status}`}>{job.status}</span>}</div>
        {!result && <>{mapArea ? <GeographicMap area={mapArea} /> : <div className="map-loading">{custom ? "Validate the custom bounds to confirm and map the next analysis area." : "Loading the backend-validated Alappuzha boundary…"}</div>}{!custom && preset?.preview ? <figure className="saved-preview"><img src={asset(preset.preview.image_url)} alt={`Previous true-colour observation from ${preset.preview.acquisition_date}`} /><figcaption><span><strong>Previous saved observation · {preset.preview.acquisition_date}</strong>{preset.preview.label}</span><DownloadLink href={preset.preview.source_record_url}>Source record</DownloadLink></figcaption></figure> : !custom && preset && <div className="preview-unavailable"><Icon name="info" /><span>{preset.preview_unavailable_reason}</span></div>}</>}
        {result && <><div className="year-strip">{result.observations.map((item: any) => <button key={item.year} disabled={item.status !== "available"} className={observation?.year === item.year ? "active" : ""} onClick={() => setSelectedYear(item.year)}><span>{item.year}</span><small>{item.status === "available" ? item.acquisition_date : "Unavailable"}</small></button>)}</div>{observation && <div className="side-view real-images"><figure><img src={asset(observation.true_color_url)} alt={`True-colour observation ${observation.year}`} /><figcaption>{observation.acquisition_date} · true colour</figcaption></figure><figure><img src={asset(observation.ndvi_url)} alt={`NDVI ${observation.year}`} /><figcaption>{observation.acquisition_date} · NDVI</figcaption></figure></div>}<h3>NDVI differences</h3><div className="interval-tabs">{differences.map((item: any) => <button key={item.label} className={difference?.label === item.label ? "active" : ""} onClick={() => setSelectedDifference(item.label)}>{item.before_year} → {item.after_year}</button>)}</div>{difference?.status === "available" ? <figure className="overlay-figure"><img src={asset(difference.difference_url)} alt={`NDVI difference ${difference.before_year} to ${difference.after_year}`} /><figcaption>Blue: NDVI increased · Red: decreased · White: little difference · Grey: excluded</figcaption></figure> : difference && <div className="caution"><Icon name="info" /><span>{difference.reason}</span></div>}</>}
      </section>
    </div>
  </main>
}

function Welcome({ onStart }: { onStart: (view: View) => void }) {
  const [provider, setProvider] = useState<any>(null)
  useEffect(() => { api<any>("/api/provider").then(setProvider).catch(() => setProvider({ available: false })) }, [])
  return <main className="welcome"><header className="topbar"><span className={`live-badge ${provider?.available ? "" : "warning"}`}><span />{provider === null ? "Checking local services" : provider.available ? `${provider.provider}/${provider.model} ready` : "Local model unavailable"}</span></header><section className="welcome-content"><div className="eyebrow"><Icon name="spark" /> Evidence-first local analysis</div><h1>See how places change.</h1><p className="welcome-copy">Run the existing GeoLens analysis tools through a separate local frontend. Every answer links back to saved imagery, records and explicit limitations.</p><div className="welcome-actions"><button className="primary-button" onClick={() => onStart("infrastructure")}><Icon name="building" /> Buildings and infrastructure</button><button className="secondary-button" onClick={() => onStart("vegetation")}><Icon name="leaf" /> Vegetation history</button></div><div className="trust-row"><span><Icon name="check" /> Real local execution</span><span><Icon name="check" /> Explicit Run action</span><span><Icon name="check" /> Streamlit fallback preserved</span></div></section></main>
}

export default function App() {
  const [view, setView] = useState<View>("welcome")
  const content = useMemo(() => view === "infrastructure" ? <Infrastructure /> : view === "vegetation" ? <Vegetation /> : <Welcome onStart={setView} />, [view])
  return <div className="app-shell"><Sidebar view={view} onNavigate={setView} />{content}</div>
}
