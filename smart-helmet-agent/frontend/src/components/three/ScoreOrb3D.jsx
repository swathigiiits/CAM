import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Float, Torus } from '@react-three/drei'

export function ScoreOrb3D({ score = 75 }) {
  const ring = useRef()
  const pct = Math.min(1, Math.max(0, score / 100))
  const color = score >= 75 ? '#22d3ee' : score >= 55 ? '#fbbf24' : '#f87171'

  useFrame((state) => {
    if (ring.current) ring.current.rotation.z = state.clock.elapsedTime * 0.4
  })

  return (
    <Float speed={2} floatIntensity={0.3}>
      <group>
        <Torus ref={ring} args={[1.1, 0.04, 16, 100]} rotation={[Math.PI / 2, 0, 0]}>
          <meshStandardMaterial color="#1e293b" metalness={0.5} roughness={0.5} />
        </Torus>
        <Torus
          args={[1.1, 0.07, 16, 100, pct * Math.PI * 2]}
          rotation={[Math.PI / 2, 0, -Math.PI / 2]}
        >
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={0.9}
            metalness={0.9}
            roughness={0.1}
          />
        </Torus>
        <mesh>
          <sphereGeometry args={[0.55, 32, 32]} />
          <meshStandardMaterial
            color="#0f172a"
            emissive={color}
            emissiveIntensity={0.25}
            metalness={0.8}
            roughness={0.2}
          />
        </mesh>
      </group>
    </Float>
  )
}
