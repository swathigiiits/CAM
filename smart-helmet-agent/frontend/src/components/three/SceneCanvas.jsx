import { Suspense } from 'react'
import { Canvas } from '@react-three/fiber'

export function SceneCanvas({
  children,
  className = '',
  camera = { position: [0, 0, 5.5], fov: 42 },
  interactive = false,
}) {
  return (
    <div className={`${className} ${interactive ? '' : 'pointer-events-none'}`}>
      <Canvas
        camera={camera}
        dpr={[1, 1.75]}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        style={{ background: 'transparent' }}
      >
        <Suspense fallback={null}>{children}</Suspense>
      </Canvas>
    </div>
  )
}
