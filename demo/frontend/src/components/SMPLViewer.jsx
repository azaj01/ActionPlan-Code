import React, { useRef, useMemo, useEffect, useState } from 'react'
import { Canvas, useThree, useFrame } from '@react-three/fiber'
import { OrbitControls, Sky, PerspectiveCamera, Outlines } from '@react-three/drei'
import * as THREE from 'three'

const ROOM_SIZE = 60
const WALL_HEIGHT = 20

function toFlatVertices(vertices) {
  if (!vertices) return null
  return vertices instanceof Float32Array ? vertices : vertices.flat()
}

function getFloorY(vertices) {
  if (!vertices?.length) return 0
  const flat = toFlatVertices(vertices)
  let minY = Infinity
  for (let i = 1; i < flat.length; i += 3) minY = Math.min(minY, flat[i])
  return minY === Infinity ? 0 : minY - 0.01
}

function getFloorYFromMeshes(meshes) {
  if (!meshes?.length) return 0
  let minY = Infinity
  for (const m of meshes) {
    if (m.vertices?.length) {
      const flat = toFlatVertices(m.vertices)
      for (let i = 1; i < flat.length; i += 3) minY = Math.min(minY, flat[i])
    }
  }
  return minY === Infinity ? 0 : minY - 0.01
}

function SMPLMesh({ vertices, faces, meshColor, showOutline, transparent }) {
  const meshRef = useRef()

  const geometry = useMemo(() => {
    if (!vertices || !faces) return null

    const geom = new THREE.BufferGeometry()

    const flat = toFlatVertices(vertices)
    const positions = flat instanceof Float32Array ? flat : new Float32Array(flat)
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3))

    const indices = new Uint32Array(faces.flat())
    geom.setIndex(new THREE.BufferAttribute(indices, 1))

    geom.computeVertexNormals()

    return geom
  }, [faces])

  useEffect(() => {
    if (meshRef.current && vertices) {
      const geom = meshRef.current.geometry
      const positions = geom.attributes.position
      const flatVertices = toFlatVertices(vertices)

      for (let i = 0; i < flatVertices.length; i++) {
        positions.array[i] = flatVertices[i]
      }
      positions.needsUpdate = true

      geom.computeVertexNormals()
      geom.attributes.normal.needsUpdate = true
    }
  }, [vertices])

  if (!geometry) return null

  const matProps = transparent
    ? { transparent: true, opacity: 0.28, depthWrite: false }
    : {}

  return (
    <mesh ref={meshRef} geometry={geometry} castShadow={!transparent} receiveShadow={!transparent}>
      <meshStandardMaterial
        color={meshColor ?? '#8ea6ff'}
        roughness={0.82}
        metalness={0}
        side={THREE.FrontSide}
        envMapIntensity={0}
        {...matProps}
      />
      {showOutline && <Outlines thickness={0.08} color="#1a1a2e" />}
    </mesh>
  )
}

function getMeshBounds(vertices) {
  if (!vertices?.length) return { center: [0, 1, 0], height: 1.7 }
  const flat = toFlatVertices(vertices)
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity
  for (let i = 0; i < flat.length; i += 3) {
    minX = Math.min(minX, flat[i]); maxX = Math.max(maxX, flat[i])
    minY = Math.min(minY, flat[i + 1]); maxY = Math.max(maxY, flat[i + 1])
    minZ = Math.min(minZ, flat[i + 2]); maxZ = Math.max(maxZ, flat[i + 2])
  }
  const center = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2]
  const height = maxY - minY
  return { center, height }
}

const HEADING_WINDOW = 14
const HEADING_UNLOCK_SPEED = 0.03
const HEADING_LOCK_SPEED = 0.015
const HEADING_MIN_NET_DISP = 0.09
const HEADING_COHERENCE_MIN = 0.55
const MAX_YAW_RATE = 0.85 // rad/s
const USER_INTERACT_TIMEOUT = 3.0 // seconds before auto-follow resumes

function shortestAngleDelta(target, current) {
  return Math.atan2(Math.sin(target - current), Math.cos(target - current))
}

function CameraAndFollow({ vertices, controlsRef }) {
  const { camera, gl } = useThree()
  const initialized = useRef(false)
  const prevCenterRef = useRef(new THREE.Vector3())
  const headingSamplesRef = useRef([])
  const yawRef = useRef(0)
  const headingLockedRef = useRef(true)
  const userControllingRef = useRef(false)
  const lastInteractTimeRef = useRef(0)

  useEffect(() => {
    const canvas = gl.domElement
    const onStart = () => {
      userControllingRef.current = true
      lastInteractTimeRef.current = performance.now() / 1000
    }
    const onEnd = () => {
      lastInteractTimeRef.current = performance.now() / 1000
    }
    canvas.addEventListener('pointerdown', onStart)
    canvas.addEventListener('pointerup', onEnd)
    canvas.addEventListener('wheel', () => {
      userControllingRef.current = true
      lastInteractTimeRef.current = performance.now() / 1000
    })
    return () => {
      canvas.removeEventListener('pointerdown', onStart)
      canvas.removeEventListener('pointerup', onEnd)
    }
  }, [gl])

  useFrame((_, delta) => {
    const controls = controlsRef?.current
    if (!vertices || !controls) return

    const now = performance.now() / 1000
    if (userControllingRef.current && now - lastInteractTimeRef.current > USER_INTERACT_TIMEOUT) {
      userControllingRef.current = false
    }

    const { center, height } = getMeshBounds(vertices)
    const cx = center[0]
    const cy = center[1]
    const cz = center[2]

    if (!initialized.current) {
      const dist = Math.max(height * 2.6, 3.5)
      const camHeight = cy + height * 0.4
      camera.position.set(cx, camHeight, cz + dist)
      camera.lookAt(cx, cy, cz)
      prevCenterRef.current.set(cx, cy, cz)
      headingSamplesRef.current = [new THREE.Vector2(cx, cz)]
      yawRef.current = 0
      controls.target.set(cx, cy, cz)
      initialized.current = true
      return
    }

    const smoothCx = (cx + prevCenterRef.current.x) * 0.5
    const smoothCy = (cy + prevCenterRef.current.y) * 0.5
    const smoothCz = (cz + prevCenterRef.current.z) * 0.5

    const samples = headingSamplesRef.current
    samples.push(new THREE.Vector2(smoothCx, smoothCz))
    if (samples.length > HEADING_WINDOW) samples.shift()

    const vx = cx - prevCenterRef.current.x
    const vz = cz - prevCenterRef.current.z
    const speed = Math.sqrt(vx * vx + vz * vz)

    if (headingLockedRef.current) {
      if (speed > HEADING_UNLOCK_SPEED) headingLockedRef.current = false
    } else if (speed < HEADING_LOCK_SPEED) {
      headingLockedRef.current = true
    }

    if (!headingLockedRef.current && samples.length >= 2) {
      const start = samples[0]
      const end = samples[samples.length - 1]
      const hx = end.x - start.x
      const hz = end.y - start.y
      const hLen = Math.sqrt(hx * hx + hz * hz)
      let pathLen = 0
      for (let i = 1; i < samples.length; i++) {
        const dx = samples[i].x - samples[i - 1].x
        const dz = samples[i].y - samples[i - 1].y
        pathLen += Math.sqrt(dx * dx + dz * dz)
      }
      const coherence = pathLen > 1e-4 ? hLen / pathLen : 1

      if (hLen > HEADING_MIN_NET_DISP && coherence > HEADING_COHERENCE_MIN) {
        const targetYaw = Math.atan2(hx / hLen, hz / hLen)
        const maxStep = MAX_YAW_RATE * Math.max(delta, 1 / 120)
        const dYaw = shortestAngleDelta(targetYaw, yawRef.current)
        const clamped = Math.max(-maxStep, Math.min(maxStep, dYaw))
        yawRef.current += clamped
      }
    }

    prevCenterRef.current.set(cx, cy, cz)

    if (userControllingRef.current) return

    const dist = Math.max(height * 2.6, 3.5)
    const camHeight = smoothCy + height * 0.4
    const fx = Math.sin(yawRef.current)
    const fz = Math.cos(yawRef.current)
    camera.position.set(
      smoothCx + fx * dist,
      camHeight,
      smoothCz + fz * dist
    )
    controls.target.set(smoothCx, smoothCy, smoothCz)
  }, -2)
  return null
}

function Room({ floorY }) {
  const half = ROOM_SIZE / 2
  const wallCenterY = floorY + WALL_HEIGHT / 2
  const { wallMap, wallBumpMap } = useMemo(() => {
    const size = 1024
    const panelsX = 8
    const panelsY = 4
    const panelW = size / panelsX
    const panelH = size / panelsY
    const inset = 10

    const colorCanvas = document.createElement('canvas')
    colorCanvas.width = size
    colorCanvas.height = size
    const colorCtx = colorCanvas.getContext('2d')

    colorCtx.fillStyle = '#e8ecf2'
    colorCtx.fillRect(0, 0, size, size)
    colorCtx.strokeStyle = '#c8d0dc'
    colorCtx.lineWidth = 1

    for (let y = 0; y < panelsY; y++) {
      for (let x = 0; x < panelsX; x++) {
        const px = x * panelW + inset
        const py = y * panelH + inset
        const pw = panelW - inset * 2
        const ph = panelH - inset * 2

        const grad = colorCtx.createLinearGradient(px, py, px + pw, py + ph)
        grad.addColorStop(0, '#f5f7fa')
        grad.addColorStop(0.5, '#ebeef4')
        grad.addColorStop(1, '#dfe4ec')
        colorCtx.fillStyle = grad
        colorCtx.fillRect(px, py, pw, ph)
        colorCtx.strokeRect(px, py, pw, ph)
      }
    }

    const bumpCanvas = document.createElement('canvas')
    bumpCanvas.width = size
    bumpCanvas.height = size
    const bumpCtx = bumpCanvas.getContext('2d')

    bumpCtx.fillStyle = '#7f7f7f'
    bumpCtx.fillRect(0, 0, size, size)

    for (let y = 0; y < panelsY; y++) {
      for (let x = 0; x < panelsX; x++) {
        const px = x * panelW + inset
        const py = y * panelH + inset
        const pw = panelW - inset * 2
        const ph = panelH - inset * 2

        bumpCtx.fillStyle = '#8a8a8a'
        bumpCtx.fillRect(px, py, pw, ph)
        bumpCtx.strokeStyle = '#6d6d6d'
        bumpCtx.lineWidth = 6
        bumpCtx.strokeRect(px, py, pw, ph)
      }
    }

    const map = new THREE.CanvasTexture(colorCanvas)
    map.colorSpace = THREE.SRGBColorSpace
    map.wrapS = THREE.RepeatWrapping
    map.wrapT = THREE.RepeatWrapping
    map.repeat.set(3.2, 1.2)

    const bumpMap = new THREE.CanvasTexture(bumpCanvas)
    bumpMap.wrapS = THREE.RepeatWrapping
    bumpMap.wrapT = THREE.RepeatWrapping
    bumpMap.repeat.set(3.2, 1.2)

    return { wallMap: map, wallBumpMap: bumpMap }
  }, [])

  useEffect(() => {
    return () => {
      wallMap.dispose()
      wallBumpMap.dispose()
    }
  }, [wallMap, wallBumpMap])

  const wallMaterial = {
    color: '#f8f9fc',
    roughness: 0.95,
    metalness: 0.03,
    map: wallMap,
    bumpMap: wallBumpMap,
    bumpScale: 0.22,
    side: THREE.DoubleSide,
  }
  return (
    <group>
      {/* Floor at foot level */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, floorY, 0]} receiveShadow>
        <planeGeometry args={[ROOM_SIZE, ROOM_SIZE]} />
        <meshStandardMaterial color="#d4d2cc" roughness={0.9} metalness={0} side={THREE.DoubleSide} />
      </mesh>
      {/* Subtle floor grid - 4 divisions for visible center cross */}
      <gridHelper
        args={[ROOM_SIZE, 4, '#b4b0a8', '#c4c0b8']}
        position={[0, floorY + 0.02, 0]}
      />
      {/* Back wall */}
      <mesh position={[0, wallCenterY, -half]} receiveShadow>
        <planeGeometry args={[ROOM_SIZE, WALL_HEIGHT]} />
        <meshStandardMaterial {...wallMaterial} />
      </mesh>
      {/* Front wall (fourth wall) */}
      <mesh position={[0, wallCenterY, half]} receiveShadow>
        <planeGeometry args={[ROOM_SIZE, WALL_HEIGHT]} />
        <meshStandardMaterial {...wallMaterial} />
      </mesh>
      {/* Left wall */}
      <mesh position={[-half, wallCenterY, 0]} rotation={[0, Math.PI / 2, 0]} receiveShadow>
        <planeGeometry args={[ROOM_SIZE, WALL_HEIGHT]} />
        <meshStandardMaterial {...wallMaterial} />
      </mesh>
      {/* Right wall */}
      <mesh position={[half, wallCenterY, 0]} rotation={[0, -Math.PI / 2, 0]} receiveShadow>
        <planeGeometry args={[ROOM_SIZE, WALL_HEIGHT]} />
        <meshStandardMaterial {...wallMaterial} />
      </mesh>
    </group>
  )
}

function getCombinedBounds(meshes) {
  if (!meshes?.length) return { center: [0, 1, 0], height: 1.7 }
  const flat = (v) => (v instanceof Float32Array ? v : v.flat())
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity
  for (const m of meshes) {
    if (!m.vertices?.length) continue
    const v = flat(m.vertices)
    for (let i = 0; i < v.length; i += 3) {
      minX = Math.min(minX, v[i]); maxX = Math.max(maxX, v[i])
      minY = Math.min(minY, v[i + 1]); maxY = Math.max(maxY, v[i + 1])
      minZ = Math.min(minZ, v[i + 2]); maxZ = Math.max(maxZ, v[i + 2])
    }
  }
  const center = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2]
  const height = maxY - minY
  return { center, height }
}

function Scene({ vertices, faces, meshColor, meshes, fixedCamera }) {
  const controlsRef = useRef()
  const [frozenFloorY, setFrozenFloorY] = useState(null)
  const initialTarget = useMemo(() => {
    if (meshes?.length) {
      const { center } = getCombinedBounds(meshes)
      return center
    }
    if (vertices?.length) {
      const { center } = getMeshBounds(vertices)
      return center
    }
    return [0, 1, 0]
  }, [vertices?.length, meshes?.length])

  useEffect(() => {
    if (frozenFloorY !== null) return
    if (meshes?.length) {
      setFrozenFloorY(getFloorYFromMeshes(meshes))
      return
    }
    if (vertices?.length) {
      setFrozenFloorY(getFloorY(vertices))
    }
  }, [frozenFloorY, meshes, vertices])

  const floorY = frozenFloorY ?? (meshes?.length ? getFloorYFromMeshes(meshes) : getFloorY(vertices))

  return (
    <>
      <color attach="background" args={['#8ec7ff']} />

      <Sky
        distance={450}
        sunPosition={[35, 120, -30]}
        inclination={0.5}
        azimuth={0.25}
        turbidity={2.2}
        rayleigh={3}
        mieCoefficient={0.002}
        mieDirectionalG={0.88}
      />

      <PerspectiveCamera makeDefault position={[-9.5, 2.8, 4]} fov={50} />

      <ambientLight intensity={0.55} />
      <hemisphereLight args={['#e8f0f8', '#d4ccc4', 0.5]} />
      <directionalLight
        position={[8, 12, 10]}
        intensity={0.85}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-camera-far={120}
        shadow-camera-left={-20}
        shadow-camera-right={20}
        shadow-camera-top={20}
        shadow-camera-bottom={-20}
      />
      <directionalLight position={[-6, 8, -8]} intensity={0.4} />

      {meshes?.length > 0 &&
        meshes.map((m, i) => (
          <SMPLMesh
            key={i}
            vertices={m.vertices}
            faces={faces}
            meshColor={m.color}
            showOutline={m.showOutline ?? true}
            transparent={m.transparent ?? true}
          />
        ))}
      {vertices?.length > 0 && (
        <SMPLMesh vertices={vertices} faces={faces} meshColor={meshColor} showOutline={false} transparent={false} />
      )}

      <Room floorY={floorY} />

      {!fixedCamera && <CameraAndFollow vertices={vertices} controlsRef={controlsRef} />}

      <OrbitControls
        ref={controlsRef}
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        enableDamping={false}
        minDistance={0.8}
        maxDistance={80}
        target={initialTarget}
      />
    </>
  )
}

function CameraIcon({ manual }) {
  if (manual) {
    return (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2v6M9 5l3-3 3 3" />
        <path d="M12 22v-6M9 19l3 3 3-3" />
        <path d="M2 12h6M5 9l-3 3 3 3" />
        <path d="M22 12h-6M19 9l3 3-3 3" />
      </svg>
    )
  }
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
      <circle cx="12" cy="13" r="4" />
    </svg>
  )
}

function SMPLViewer({ vertices, faces, meshColor, meshes, fixedCamera: fixedCameraProp }) {
  const [manualCamera, setManualCamera] = useState(!!fixedCameraProp)

  return (
    <div className="smpl-viewer-wrapper">
      <button
        type="button"
        className="camera-toggle-btn"
        onClick={() => setManualCamera((m) => !m)}
        title={manualCamera ? 'Resume auto-follow camera' : 'Take manual camera control'}
        aria-label={manualCamera ? 'Resume auto-follow camera' : 'Take manual camera control'}
      >
        <CameraIcon manual={manualCamera} />
      </button>
      <Canvas
        shadows
        dpr={[1, 2]}
        gl={{ antialias: true }}
        style={{ width: '100%', height: '100%' }}
      >
        <Scene
          vertices={vertices}
          faces={faces}
          meshColor={meshColor}
          meshes={meshes}
          fixedCamera={manualCamera}
        />
      </Canvas>
    </div>
  )
}

export default SMPLViewer
