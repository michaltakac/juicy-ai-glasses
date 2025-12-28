import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

interface CameraStatus {
  connected: boolean
  resolution: string
  fps: number
  backend: string
}

interface TestApp {
  id: string
  name: string
  description: string
  running: boolean
}

interface Detection {
  class_id: number
  class_name: string
  confidence: number
  bbox: [number, number, number, number]
}

interface DetectorStatus {
  enabled: boolean
  model: string | null
  classes: string[]
  total_classes: number
}

const RESOLUTIONS = [
  { label: '4K (3840×2160)', width: 3840, height: 2160 },
  { label: '1080p (1920×1080)', width: 1920, height: 1080 },
  { label: '720p (1280×720)', width: 1280, height: 720 },
  { label: '480p (640×480)', width: 640, height: 480 },
  { label: 'Square (1024×1024)', width: 1024, height: 1024 },
]

const TEST_APPS: TestApp[] = [
  { id: 'what_am_i_seeing', name: 'What Am I Seeing?', description: 'Describe current view using LLM', running: false },
  { id: 'what_am_i_wearing', name: 'What Am I Wearing?', description: 'Analyze clothing and colors', running: false },
  { id: 'object_detection', name: 'Object Detection', description: 'YOLO detection via Hailo AI HAT+', running: false },
]

// Backend URL - use direct URL for MJPEG stream to bypass Vite proxy issues
const BACKEND_URL = 'http://rpi5kavecany.local:8765'

function App() {
  const [status, setStatus] = useState<CameraStatus>({
    connected: false,
    resolution: '1920x1080',
    fps: 0,
    backend: 'unknown',
  })
  const [selectedRes, setSelectedRes] = useState(RESOLUTIONS[1])
  const [streamUrl, setStreamUrl] = useState(`${BACKEND_URL}/stream`)
  const [fpsHistory, setFpsHistory] = useState<number[]>([])
  const [testApps, setTestApps] = useState(TEST_APPS)
  const [lastResult, setLastResult] = useState<string | null>(null)
  const [isCapturing, setIsCapturing] = useState(false)
  const [detectorStatus, setDetectorStatus] = useState<DetectorStatus | null>(null)
  const [liveDetections, setLiveDetections] = useState<Detection[]>([])
  const [isDetecting, setIsDetecting] = useState(false)
  const frameCountRef = useRef(0)
  const lastTimeRef = useRef(Date.now())
  const imgRef = useRef<HTMLImageElement>(null)

  // FPS calculation
  const onFrame = useCallback(() => {
    frameCountRef.current++
    const now = Date.now()
    const elapsed = now - lastTimeRef.current
    
    if (elapsed >= 1000) {
      const fps = Math.round((frameCountRef.current * 1000) / elapsed)
      setFpsHistory(prev => [...prev.slice(-59), fps])
      setStatus(prev => ({ ...prev, fps }))
      frameCountRef.current = 0
      lastTimeRef.current = now
    }
  }, [])

  // Check camera status
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch('/api/status')
        if (res.ok) {
          const data = await res.json()
          setStatus(prev => ({ ...prev, ...data, connected: true }))
        }
      } catch {
        setStatus(prev => ({ ...prev, connected: false }))
      }
    }
    
    checkStatus()
    const interval = setInterval(checkStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  // Check detector status
  useEffect(() => {
    const checkDetector = async () => {
      try {
        const res = await fetch('/api/detector')
        if (res.ok) {
          const data = await res.json()
          setDetectorStatus(data)
        }
      } catch {
        setDetectorStatus(null)
      }
    }
    
    checkDetector()
  }, [])

  // Update stream URL when resolution changes
  useEffect(() => {
    setStreamUrl(`${BACKEND_URL}/stream?width=${selectedRes.width}&height=${selectedRes.height}&t=${Date.now()}`)
  }, [selectedRes])

  // Handle image load for FPS
  useEffect(() => {
    const img = imgRef.current
    if (!img) return

    const handleLoad = () => {
      onFrame()
      // Force reload for MJPEG stream FPS tracking
      if (img.src.includes('/stream')) {
        // MJPEG handles its own streaming
      }
    }

    img.addEventListener('load', handleLoad)
    return () => img.removeEventListener('load', handleLoad)
  }, [onFrame])

  const changeResolution = async (res: typeof RESOLUTIONS[0]) => {
    setSelectedRes(res)
    try {
      await fetch(`/api/resolution?width=${res.width}&height=${res.height}`, { method: 'POST' })
    } catch (e) {
      console.error('Failed to change resolution:', e)
    }
  }

  const runTestApp = async (appId: string) => {
    setTestApps(apps => apps.map(a => 
      a.id === appId ? { ...a, running: true } : a
    ))
    setLastResult(null)

    try {
      const res = await fetch(`/api/run/${appId}`, { method: 'POST' })
      const data = await res.json()
      setLastResult(data.result || data.error || 'Completed')
    } catch (e) {
      setLastResult(`Error: ${e}`)
    } finally {
      setTestApps(apps => apps.map(a => 
        a.id === appId ? { ...a, running: false } : a
      ))
    }
  }

  const captureSnapshot = async () => {
    setIsCapturing(true)
    try {
      const res = await fetch('/api/capture', { method: 'POST' })
      const data = await res.json()
      setLastResult(`📸 Saved: ${data.path}`)
    } catch (e) {
      setLastResult(`Error: ${e}`)
    } finally {
      setIsCapturing(false)
    }
  }

  const runLiveDetection = async () => {
    setIsDetecting(true)
    try {
      const res = await fetch('/api/detect')
      const data = await res.json()
      setLiveDetections(data.detections || [])
      if (data.detections?.length > 0) {
        const summary = data.detections.slice(0, 5).map((d: Detection) => 
          `${d.class_name} (${Math.round(d.confidence * 100)}%)`
        ).join(', ')
        setLastResult(`🔍 Found ${data.count} objects in ${data.inference_ms}ms: ${summary}`)
      } else {
        setLastResult(`🔍 No objects detected (${data.inference_ms}ms)`)
      }
    } catch (e) {
      setLastResult(`Error: ${e}`)
    } finally {
      setIsDetecting(false)
    }
  }

  const avgFps = fpsHistory.length > 0 
    ? Math.round(fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length)
    : 0

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <span className="logo-icon">👓</span>
          <h1>AI Glasses</h1>
          <span className="badge">Camera Viewer</span>
        </div>
        <div className="status-bar">
          <span className={`status-dot ${status.connected ? 'connected' : 'disconnected'}`} />
          <span className="mono">{status.connected ? 'IMX500 Connected' : 'Disconnected'}</span>
        </div>
      </header>

      <main className="main">
        <div className="video-section">
          <div className="video-container">
            <img 
              ref={imgRef}
              src={streamUrl} 
              alt="Camera Stream" 
              className="video-stream"
            />
            <div className="video-overlay">
              <div className="overlay-top">
                <span className="res-badge mono">{selectedRes.width}×{selectedRes.height}</span>
                <span className={`fps-badge mono ${status.fps >= 25 ? 'good' : status.fps >= 15 ? 'ok' : 'low'}`}>
                  {status.fps} FPS
                </span>
              </div>
              <div className="overlay-bottom">
                <span className="mono">{status.backend}</span>
              </div>
            </div>
          </div>

          <div className="fps-graph">
            <div className="fps-label">
              <span>FPS History</span>
              <span className="mono">avg: {avgFps}</span>
            </div>
            <div className="fps-bars">
              {fpsHistory.map((fps, i) => (
                <div 
                  key={i} 
                  className="fps-bar" 
                  style={{ 
                    height: `${Math.min(100, (fps / 30) * 100)}%`,
                    backgroundColor: fps >= 25 ? 'var(--accent-green)' : fps >= 15 ? 'var(--accent-yellow)' : 'var(--accent-magenta)'
                  }} 
                />
              ))}
            </div>
          </div>
        </div>

        <aside className="controls">
          <section className="control-section">
            <h2>Resolution</h2>
            <div className="res-buttons">
              {RESOLUTIONS.map(res => (
                <button
                  key={res.label}
                  className={`res-btn ${selectedRes.label === res.label ? 'active' : ''}`}
                  onClick={() => changeResolution(res)}
                >
                  {res.label}
                </button>
              ))}
            </div>
          </section>

          <section className="control-section">
            <h2>Quick Actions</h2>
            <button 
              className="action-btn capture"
              onClick={captureSnapshot}
              disabled={isCapturing}
            >
              {isCapturing ? '📸 Capturing...' : '📸 Capture Snapshot'}
            </button>
          </section>

          <section className="control-section">
            <h2>Test Apps</h2>
            <div className="test-apps">
              {testApps.map(app => (
                <button
                  key={app.id}
                  className={`test-app-btn ${app.running ? 'running' : ''}`}
                  onClick={() => runTestApp(app.id)}
                  disabled={app.running}
                >
                  <span className="app-name">{app.name}</span>
                  <span className="app-desc">{app.description}</span>
                  {app.running && <span className="spinner" />}
                </button>
              ))}
            </div>
          </section>

          {lastResult && (
            <section className="control-section result">
              <h2>Last Result</h2>
              <div className="result-box mono">
                {lastResult}
              </div>
            </section>
          )}

          <section className="control-section">
            <h2>Live Detection</h2>
            <button 
              className="action-btn detect"
              onClick={runLiveDetection}
              disabled={isDetecting || !detectorStatus?.enabled}
            >
              {isDetecting ? '🔍 Detecting...' : '🔍 Run Detection'}
            </button>
            {liveDetections.length > 0 && (
              <div className="detection-list">
                {liveDetections.map((det, i) => (
                  <div key={i} className="detection-item">
                    <span className="det-class">{det.class_name}</span>
                    <span className="det-conf">{Math.round(det.confidence * 100)}%</span>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="control-section info">
            <h2>Hardware Info</h2>
            <dl className="info-list">
              <dt>Camera</dt>
              <dd className="mono">Sony IMX500</dd>
              <dt>AI HAT+</dt>
              <dd className={`mono ${detectorStatus?.enabled ? 'status-ok' : 'status-warn'}`}>
                {detectorStatus?.enabled ? `Hailo-8 (${detectorStatus.model})` : 'Not available'}
              </dd>
              <dt>Classes</dt>
              <dd className="mono">{detectorStatus?.total_classes || 0} COCO objects</dd>
              <dt>Resolution</dt>
              <dd className="mono">{status.resolution}</dd>
            </dl>
          </section>
        </aside>
      </main>
    </div>
  )
}

export default App


