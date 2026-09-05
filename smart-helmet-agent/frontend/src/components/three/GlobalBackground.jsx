import { SceneCanvas } from './SceneCanvas.jsx'
import { AmbientScene } from './AmbientScene.jsx'

export function GlobalBackground() {
  return (
    <div className="fixed inset-0 z-0" aria-hidden>
      <SceneCanvas className="w-full h-full">
        <AmbientScene />
      </SceneCanvas>
      <div className="absolute inset-0 bg-gradient-to-b from-white/30 via-transparent to-[#f0f6ff]/80 pointer-events-none" />
      <div className="absolute inset-0 vignette pointer-events-none" />
    </div>
  )
}
