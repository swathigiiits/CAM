import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Stars, Grid } from '@react-three/drei'
import * as THREE from 'three'

function FloatingParticles({ count = 280 }) {
  const ref = useRef()
  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      arr[i * 3] = (Math.random() - 0.5) * 20
      arr[i * 3 + 1] = (Math.random() - 0.5) * 12
      arr[i * 3 + 2] = (Math.random() - 0.5) * 10
    }
    return arr
  }, [count])

  useFrame((state) => {
    if (!ref.current) return
    ref.current.rotation.y = state.clock.elapsedTime * 0.015
  })

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.04}
        color="#0c8de8"
        transparent
        opacity={0.45}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.NormalBlending}
      />
    </points>
  )
}

function ScanRing() {
  const ref = useRef()
  useFrame((state) => {
    if (ref.current) ref.current.rotation.z = state.clock.elapsedTime * 0.06
  })
  return (
    <mesh ref={ref} rotation={[Math.PI / 2.2, 0, 0]} position={[0, -1.2, -2]}>
      <ringGeometry args={[3.2, 3.35, 64]} />
      <meshBasicMaterial color="#7bc5fb" transparent opacity={0.2} side={THREE.DoubleSide} />
    </mesh>
  )
}

export function AmbientScene() {
  return (
    <>
      <color attach="background" args={['#f0f6ff']} />
      <fog attach="fog" args={['#e8f4fd', 10, 28]} />
      <ambientLight intensity={0.85} />
      <directionalLight position={[5, 8, 5]} intensity={0.9} color="#ffffff" />
      <pointLight position={[4, 3, 6]} intensity={0.5} color="#36a7f7" />
      <Stars radius={60} depth={40} count={1200} factor={2} saturation={0.2} fade speed={0.4} />
      <FloatingParticles />
      <Grid
        position={[0, -2.2, 0]}
        args={[24, 24]}
        cellSize={0.55}
        cellThickness={0.4}
        cellColor="#b9dffd"
        sectionSize={2.5}
        sectionThickness={0.8}
        sectionColor="#7bc5fb"
        fadeDistance={18}
        fadeStrength={1}
        infiniteGrid
      />
      <ScanRing />
    </>
  )
}
