import React, { useState, useCallback, useRef, useEffect } from 'react'
import SMPLViewer from './components/SMPLViewer'

// In production: use Lightning backend. In dev: use localhost (proxied via /api).
const API_BASE = import.meta.env.VITE_BACKEND_URL ||
  (import.meta.env.DEV ? 'http://localhost:8000' : 'https://8000-01kj30ae5r0yty705g9javjw46.cloudspaces.litng.ai')
const LATENT_FPS = 7.5  // matches backend latent rate
const WS_TEXT_DECODER = new TextDecoder()

function toWebSocketUrl(httpUrl) {
  if (httpUrl.startsWith('https://')) return `wss://${httpUrl.slice('https://'.length)}`
  if (httpUrl.startsWith('http://')) return `ws://${httpUrl.slice('http://'.length)}`
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = window.location.host
  const path = httpUrl.startsWith('/') ? httpUrl : `/${httpUrl}`
  return `${proto}//${host}${path}`
}

function decodeBinaryVertices(verticesBinaryBuffer) {
  const view = new DataView(verticesBinaryBuffer)
  const numFrames = view.getUint32(0, true)
  const numVertices = view.getUint32(4, true)
  const floatsPerFrame = numVertices * 3
  const floatData = new Float32Array(verticesBinaryBuffer, 8, numFrames * floatsPerFrame)
  const frames = []
  for (let i = 0; i < numFrames; i++) {
    const start = i * floatsPerFrame
    frames.push(floatData.subarray(start, start + floatsPerFrame))
  }
  return frames
}

function parseWebSocketBinaryMessage(arrayBuffer) {
  const view = new DataView(arrayBuffer)
  const headerLen = view.getUint32(0, true)
  const headerBytes = new Uint8Array(arrayBuffer, 4, headerLen)
  const header = JSON.parse(WS_TEXT_DECODER.decode(headerBytes))
  const verticesBuffer = arrayBuffer.slice(4 + headerLen)
  return { ...header, vertices_frames: decodeBinaryVertices(verticesBuffer) }
}

/** Decode base64 binary vertices (format: 8-byte header + float32 data) to array of Float32Arrays per frame */
function base64ToVertices(b64) {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  const view = new DataView(bytes.buffer)
  const numFrames = view.getUint32(0, true)
  const numVertices = view.getUint32(4, true)
  const floatsPerFrame = numVertices * 3
  const result = []
  for (let f = 0; f < numFrames; f++) {
    const start = 8 + f * floatsPerFrame * 4
    result.push(new Float32Array(bytes.buffer, start, floatsPerFrame))
  }
  return result
}

function App() {
  const [prompt, setPrompt] = useState('')
  const [duration, setDuration] = useState(5)
  const [numBlocks, setNumBlocks] = useState(1)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [progress, setProgress] = useState({ current: 0, total: 0 })
  const [progressText, setProgressText] = useState('')

  const [faces, setFaces] = useState(null)
  const [verticesCount, setVerticesCount] = useState(0)
  const [currentFrame, setCurrentFrame] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [fps, setFps] = useState(30)

  const [history, setHistory] = useState([])
  const [sessionId, setSessionId] = useState(null)

  const [samplePrompts, setSamplePrompts] = useState([])
  const [currentSamplePrompt, setCurrentSamplePrompt] = useState(null)
  const [autoGenerate, setAutoGenerate] = useState(false)
  const [frameByFrame, setFrameByFrame] = useState(false)
  const [configCollapsed, setConfigCollapsed] = useState(true)
  const [promptSectionCollapsed, setPromptSectionCollapsed] = useState(false)
  const [generatingPrompt, setGeneratingPrompt] = useState(null)

  const animationRef = useRef(null)
  const lastFrameTimeRef = useRef(0)
  const allVerticesRef = useRef([])
  const isGeneratingRef = useRef(false)
  const verticesBeforeCurrentGenRef = useRef([])
  const startFrameForThisGenRef = useRef(0)
  const historyAddedForCurrentGenRef = useRef(false)
  isGeneratingRef.current = isGenerating

  useEffect(() => {
    checkConnection()
    loadFaces()
    createSession()
  }, [])

  const loadSamplePrompts = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/sample-prompts`, { cache: 'no-store' })
      const data = await res.json()
      if (data.prompts?.length) {
        setSamplePrompts(data.prompts)
        setCurrentSamplePrompt(data.prompts[Math.floor(Math.random() * data.prompts.length)])
      }
    } catch (_e) {
      // Prompts come from backend only
    }
  }, [])

  useEffect(() => {
    loadSamplePrompts()
  }, [loadSamplePrompts])

  const pickRandomSamplePrompt = useCallback(() => {
    if (samplePrompts.length === 0) return
    setCurrentSamplePrompt(samplePrompts[Math.floor(Math.random() * samplePrompts.length)])
  }, [samplePrompts])

  const checkConnection = async () => {
    try {
      const res = await fetch(`${API_BASE}/health?t=${Date.now()}`, { cache: 'no-store' })
      const data = await res.json()
      setIsConnected(data.sampler_loaded)
    } catch (e) {
      setIsConnected(false)
    }
  }

  const loadFaces = async () => {
    try {
      const res = await fetch(`${API_BASE}/mesh/faces`, { cache: 'no-store' })
      const data = await res.json()
      setFaces(data.faces)
    } catch (e) {
      console.error('Failed to load mesh faces:', e)
    }
  }

  const createSession = async () => {
    try {
      const res = await fetch(`${API_BASE}/session/create`, { method: 'POST' })
      const data = await res.json()
      setSessionId(data.session_id)
    } catch (e) {
      console.error('Failed to create session:', e)
    }
  }

  const clearSession = async () => {
    if (sessionId) {
      try {
        await fetch(`${API_BASE}/session/${sessionId}`, { method: 'DELETE' })
      } catch (e) {
        console.error('Failed to clear session:', e)
      }
    }
    allVerticesRef.current = []
    setVerticesCount(0)
    setCurrentFrame(0)
    setHistory([])
    await createSession()
  }

  const handleGenerate = useCallback(async (overridePrompt) => {
    const textToUse = (typeof overridePrompt === 'string' ? overridePrompt : prompt).trim()
    if (!textToUse || isGenerating) return

    setIsGenerating(true)
    setGeneratingPrompt(textToUse)
    isGeneratingRef.current = true
    historyAddedForCurrentGenRef.current = false
    // Keep playback running during generation so we don't freeze at the end
    setProgress({ current: 0, total: 0 })
    setProgressText('Starting generation...')

    try {
      verticesBeforeCurrentGenRef.current = [...allVerticesRef.current]
      startFrameForThisGenRef.current = allVerticesRef.current.length
      // Don't jump playback to end; let it continue from current position

      const wsUrl = toWebSocketUrl(`${API_BASE}/ws/generate`)
      await new Promise((resolve, reject) => {
        let hadError = null
        const ws = new WebSocket(wsUrl)
        ws.binaryType = 'arraybuffer'

        ws.onopen = () => {
          ws.send(JSON.stringify({
            text: textToUse,
            seconds: duration,
            session_id: sessionId,
            num_blocks: frameByFrame ? Math.ceil(duration * LATENT_FPS) : numBlocks,
          }))
        }

        ws.onmessage = (event) => {
          try {
            const data = typeof event.data === 'string'
              ? JSON.parse(event.data)
              : parseWebSocketBinaryMessage(event.data)

            handleStreamEvent(data, { prompt: textToUse, duration })

            if (data.type === 'block_complete' && typeof data.block_idx === 'number') {
              ws.send(JSON.stringify({ type: 'ack', block_idx: data.block_idx }))
            }

            if (data.type === 'generation_complete') {
              ws.close()
            } else if (data.type === 'error') {
              hadError = data.message || 'Generation failed'
              ws.close()
            }
          } catch (err) {
            hadError = err instanceof Error ? err.message : String(err)
            ws.close()
          }
        }

        ws.onerror = () => {
          hadError = hadError || 'WebSocket error during generation'
          ws.close()
        }

        ws.onclose = () => {
          if (hadError) reject(new Error(hadError))
          else resolve()
        }
      })

      setPrompt('')
      setIsPlaying(true)

    } catch (e) {
      console.error('Generation failed:', e)
      setProgressText(`Error: ${e.message}`)
    } finally {
      setIsGenerating(false)
      setGeneratingPrompt(null)
      isGeneratingRef.current = false
      pickRandomSamplePrompt()
    }
  }, [prompt, duration, sessionId, numBlocks, frameByFrame, pickRandomSamplePrompt])

  const useSamplePrompt = useCallback(() => {
    if (currentSamplePrompt) {
      setPrompt(currentSamplePrompt)
      handleGenerate(currentSamplePrompt)
    }
  }, [currentSamplePrompt, handleGenerate])

  // When autogenerate is on and we're idle, start the next generation only after
  // the last sequence has started playing (playhead has reached its start frame)
  useEffect(() => {
    if (!autoGenerate || isGenerating || !isConnected || samplePrompts.length === 0) return

    const lastStartFrame = history.length > 0 ? (history[history.length - 1].startFrame ?? 0) : 0
    const hasReachedLastSequence = history.length === 0 || currentFrame >= lastStartFrame
    if (!hasReachedLastSequence) return

    const next = samplePrompts[Math.floor(Math.random() * samplePrompts.length)]
    setCurrentSamplePrompt(next)
    handleGenerate(next)
  }, [autoGenerate, isGenerating, isConnected, samplePrompts, handleGenerate, history, currentFrame])

  const handleStreamEvent = (data, genContext) => {
    switch (data.type) {
      case 'generation_start':
        setProgress({ current: 0, total: data.total_blocks })
        setFps(data.fps)
        setProgressText(`Generating ${data.total_new_frames} new frames in ${data.total_blocks} blocks...`)
        break

      case 'block_complete': {
        const totalBlocks = data.total_blocks ?? progress.total
        setProgress(prev => ({ ...prev, current: data.block_idx + 1, total: totalBlocks }))
        setProgressText(`Block ${data.block_idx + 1}/${totalBlocks} complete (${data.decoded_frames} frames)`)

        // Add prompt when the first newly decoded block arrives, even if block 0 was conditioning-only.
        if (!historyAddedForCurrentGenRef.current && genContext && (data.decoded_frames ?? 0) > 0) {
          historyAddedForCurrentGenRef.current = true
          setHistory(prev => [...prev, {
            prompt: genContext.prompt,
            duration: genContext.duration,
            timestamp: new Date(),
            startFrame: startFrameForThisGenRef.current,
          }])
        }

        // Accumulate: backend sends only new block vertices; append to current (prefix + prev blocks)
        const newFrames = data.vertices_frames ?? (data.vertices_b64 ? base64ToVertices(data.vertices_b64) : (data.vertices || []))
        allVerticesRef.current.push(...newFrames)
        const newLen = allVerticesRef.current.length
        setVerticesCount(newLen)

        // Clamp current frame to valid range; don't jump to start of new motion
        setCurrentFrame(prev => Math.min(prev, Math.max(0, newLen - 1)))

        // Autoplay when the first newly decoded frames arrive.
        if ((data.decoded_frames ?? 0) > 0 && !isPlaying) {
          setIsPlaying(true)
        }

        break
      }

      case 'generation_complete': {
        // WS path streams blocks incrementally, so complete may be metadata-only.
        const fullMotion = data.vertices_frames ?? (data.vertices_b64 ? base64ToVertices(data.vertices_b64) : (data.vertices ?? null))
        if (fullMotion && fullMotion.length > 0) {
          allVerticesRef.current = fullMotion
          setVerticesCount(fullMotion.length)
          setCurrentFrame(prev => Math.min(prev, Math.max(0, fullMotion.length - 1)))
        }
        setProgressText(`Complete! ${data.total_decoded_frames ?? allVerticesRef.current.length ?? 0} frames generated`)
        break
      }

      case 'error':
        setProgressText(`Error: ${data.message}`)
        break
    }
  }

  useEffect(() => {
    if (isPlaying && allVerticesRef.current.length > 0) {
      const frameInterval = 1000 / fps
      const animate = (timestamp) => {
        const len = allVerticesRef.current.length
        if (len === 0) return
        const elapsed = timestamp - lastFrameTimeRef.current
        if (elapsed >= frameInterval) {
          const framesToAdvance = Math.min(Math.floor(elapsed / frameInterval), len)
          if (framesToAdvance > 0) {
            setCurrentFrame(prev => {
              const next = Math.min(prev + framesToAdvance, len - 1)
              if (next >= len - 1) setIsPlaying(false)
              return next
            })
            // Keep fractional remainder so uneven RAF cadence does not quantize to ~20fps.
            lastFrameTimeRef.current += framesToAdvance * frameInterval
          }
        }
        animationRef.current = requestAnimationFrame(animate)
      }
      lastFrameTimeRef.current = performance.now()
      animationRef.current = requestAnimationFrame(animate)
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isPlaying, fps])

  const atEnd = verticesCount > 0 && currentFrame >= verticesCount - 1
  const togglePlayback = () => {
    if (atEnd) {
      setCurrentFrame(0)
      setIsPlaying(true)
    } else {
      setIsPlaying(prev => !prev)
    }
  }

  const handleTimelineClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const percent = x / rect.width
    setCurrentFrame(Math.floor(percent * verticesCount))
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleGenerate()
    }
  }

  const removeFromHistory = useCallback(async (index) => {
    if (index < 0 || index >= history.length || isGenerating) return
    const keepFrames = history[index].startFrame ?? 0
    try {
      const res = await fetch(`${API_BASE}/session/${sessionId}/trim`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keep_frames: keepFrames }),
      })
      if (!res.ok) throw new Error('Trim failed')
      const newVertices = allVerticesRef.current.slice(0, keepFrames)
      setHistory(prev => prev.slice(0, index))
      allVerticesRef.current = newVertices
      setVerticesCount(newVertices.length)
      setCurrentFrame(f => Math.min(f, Math.max(0, keepFrames - 1)))
    } catch (e) {
      console.error('Failed to remove from history:', e)
    }
  }, [history, sessionId, isGenerating])

  return (
    <div className="main-content">
      <div className="viewer-container">
        {faces && verticesCount > 0 ? (
          <SMPLViewer
            faces={faces}
            vertices={allVerticesRef.current[Math.min(currentFrame, verticesCount - 1)] ?? []}
          />
        ) : (
          <div className="loading-overlay">
            {!faces ? (
              <>
                <div className="spinner" />
                <div className="loading-text">Loading mesh data...</div>
              </>
            ) : (
              <div className="loading-text">
                Enter a prompt to generate motion
              </div>
            )}
          </div>
        )}
      </div>

      <div className="controls-panel">
        {!autoGenerate && (
          <div className="prompt-section-collapsible">
            <button
              type="button"
              className="prompt-section-header"
              onClick={() => setPromptSectionCollapsed(c => !c)}
            >
              <span>Text-to-Motion</span>
              <span className={`settings-chevron ${promptSectionCollapsed ? '' : 'expanded'}`}>▾</span>
            </button>
            {!promptSectionCollapsed && (
              <div className="prompt-section">
                {!isGenerating && (
                  <div className="prompt-input-container">
                    <textarea
                      className="prompt-input"
                      placeholder="Enter a prompt..."
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      onKeyDown={handleKeyDown}
                    />
                    <button
                      className="generate-btn"
                      onClick={() => handleGenerate()}
                      disabled={!prompt.trim() || !isConnected}
                    >
                      Generate
                    </button>
                  </div>
                )}
                <div className="sample-prompt-section">
                  <p className="sample-prompt-label">{isGenerating ? 'Generating:' : 'Sample prompt:'}</p>
                  <div className="sample-prompt-display">{generatingPrompt ?? currentSamplePrompt ?? '—'}</div>
                  {!isGenerating && (
                    <div className="sample-prompt-buttons">
                      <button
                        type="button"
                        className="sample-prompt-btn use-btn"
                        onClick={useSamplePrompt}
                        disabled={!currentSamplePrompt || !isConnected}
                      >
                        Use this prompt
                      </button>
                      <button
                        type="button"
                        className="sample-prompt-btn shuffle-btn"
                        onClick={pickRandomSamplePrompt}
                        title="Pick another random prompt"
                      >
                        Shuffle
                      </button>
                    </div>
                  )}
                  <label className="autogenerate-toggle autogenerate-toggle-inline">
                    <input
                      type="checkbox"
                      checked={autoGenerate}
                      onChange={(e) => setAutoGenerate(e.target.checked)}
                      disabled={!isConnected}
                    />
                    <span className="autogenerate-toggle-track">
                      <span className="autogenerate-toggle-thumb" />
                    </span>
                    <span className="autogenerate-label">Autogenerate</span>
                  </label>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="settings-section settings-section-collapsible">
          <button
            type="button"
            className="settings-section-header"
            onClick={() => setConfigCollapsed(c => !c)}
          >
            <span>Config</span>
            <span className={`settings-chevron ${configCollapsed ? '' : 'expanded'}`}>▾</span>
          </button>
          {!configCollapsed && (
            <div className="settings-section-body">
              <div className="setting-row">
                <span className="setting-label">Duration (seconds)</span>
                <input
                  type="number"
                  className="setting-input"
                  value={duration}
                  onChange={(e) => setDuration(Math.max(1, Math.min(10, Number(e.target.value))))}
                  min={1}
                  max={10}
                />
              </div>
              <div className="setting-block">
                <div className="setting-row">
                  <span className="setting-label">Streaming blocks</span>
                  <input
                    type="number"
                    className="setting-input"
                    value={frameByFrame ? Math.ceil(duration * LATENT_FPS) : numBlocks}
                    onChange={(e) => setNumBlocks(Math.max(1, Math.min(20, Number(e.target.value) || 1)))}
                    min={1}
                    max={20}
                    disabled={frameByFrame}
                  />
                </div>
                <label className="setting-checkbox" title="Send latents one by one (1 latent per block)">
                  <input
                    type="checkbox"
                    checked={frameByFrame}
                    onChange={(e) => setFrameByFrame(e.target.checked)}
                  />
                  <span>Frame by frame</span>
                </label>
              </div>
              <div className="setting-row">
                <span className="setting-label">Session</span>
                <button className="clear-btn" onClick={clearSession}>
                  Clear & Reset
                </button>
              </div>
            </div>
          )}
        </div>

        {autoGenerate && (
          <div className={`autogenerate-bar ${!isConnected ? 'autogenerate-bar-disabled' : ''}`}>
            <label className="autogenerate-toggle">
              <input
                type="checkbox"
                checked={autoGenerate}
                onChange={(e) => setAutoGenerate(e.target.checked)}
                disabled={!isConnected}
              />
              <span className="autogenerate-toggle-track">
                <span className="autogenerate-toggle-thumb" />
              </span>
              <span className="autogenerate-label">Autogenerate</span>
            </label>
          </div>
        )}

        {isGenerating && (
          <div className="progress-bar-container">
            <div className="progress-bar">
              <div
                className="progress-bar-fill"
                style={{ width: `${progress.total ? (progress.current / progress.total) * 100 : 0}%` }}
              />
            </div>
          </div>
        )}

        <div className="history-section">
          <h3>Generation History</h3>
          {history.length === 0 ? (
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
              No generations yet
            </p>
          ) : (
            history.map((item, i) => {
              const startFrame = item.startFrame ?? 0
              const endFrame = i + 1 < history.length ? (history[i + 1].startFrame ?? verticesCount) : verticesCount
              const isActive = currentFrame >= startFrame && currentFrame < endFrame
              const showTracer = isActive && isPlaying
              return (
                <div
                  key={i}
                  className={`history-item history-item-clickable ${showTracer ? 'history-item-active' : !isActive ? 'history-item-inactive' : ''}`}
                  title="Click to jump to this motion"
                  onClick={() => {
                    const frame = item.startFrame ?? 0
                    setCurrentFrame(frame)
                  }}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      const frame = item.startFrame ?? 0
                      setCurrentFrame(frame)
                    }
                  }}
                >
                  <div className="history-item-content">
                    <p>{item.prompt}</p>
                    <span>{item.duration}s · starts at frame {item.startFrame ?? 0}</span>
                  </div>
                  <button
                    type="button"
                    className="history-item-remove"
                    onClick={(e) => {
                      e.stopPropagation()
                      removeFromHistory(i)
                    }}
                    disabled={isGenerating}
                    title="Remove this and all later generations"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="3 6 5 6 21 6" />
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                      <line x1="10" y1="11" x2="10" y2="17" />
                      <line x1="14" y1="11" x2="14" y2="17" />
                    </svg>
                  </button>
                </div>
              );
            })
          )}
        </div>

        {verticesCount > 0 && (
          <div className="playback-controls">
            <button className="play-btn" onClick={togglePlayback} title={atEnd ? 'Replay' : (isPlaying ? 'Pause' : 'Play')}>
              {isPlaying ? '⏸' : atEnd ? '↻' : '▶'}
            </button>
            <div className="timeline" onClick={handleTimelineClick}>
              <div
                className="timeline-progress"
                style={{ width: `${(currentFrame / Math.max(1, verticesCount - 1)) * 100}%` }}
              />
            </div>
            <div className="frame-counter">
              {currentFrame + 1} / {verticesCount}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
