import { useRef, useEffect } from 'react'

const CANVAS_BG = '#06090f'

const BLOB_CONFIGS = [
  { cx: 0.15, cy: 0.25, radius: 0.28, color: [18, 171, 219] as [number,number,number],  alpha: 0.15, speed: 0.00009,  orbitX: 0.08, orbitY: 0.07, phase: 0   },
  { cx: 0.75, cy: 0.70, radius: 0.25, color: [90, 60, 180]  as [number,number,number],  alpha: 0.12, speed: 0.000075, orbitX: 0.07, orbitY: 0.08, phase: 2.0 },
  { cx: 0.50, cy: 0.10, radius: 0.22, color: [140, 50, 120] as [number,number,number],  alpha: 0.09, speed: 0.00006,  orbitX: 0.06, orbitY: 0.09, phase: 4.0 },
  { cx: 0.90, cy: 0.30, radius: 0.18, color: [18, 140, 200] as [number,number,number],  alpha: 0.08, speed: 0.00010,  orbitX: 0.05, orbitY: 0.06, phase: 1.0 },
  { cx: 0.30, cy: 0.85, radius: 0.20, color: [180, 40, 100] as [number,number,number],  alpha: 0.07, speed: 0.00011,  orbitX: 0.07, orbitY: 0.05, phase: 3.0 },
]

const CONSTELLATION_RGB = '18,171,219'
const MAX_PULSES = 25
const PULSE_INTERVAL = 250
const EDGE_MAX_DIST_FRAC = 0.22
const MOUSE_REPEL_RADIUS = 100
const MOUSE_REPEL_FORCE = 1.0

interface AnimatedBackgroundProps {
  className?: string
}

export default function AnimatedBackground({ className }: AnimatedBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const canvas = canvasRef.current!
    if (!canvas) return
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const ctx = canvas.getContext('2d')!
    if (!ctx) return

    const dpr = Math.min(window.devicePixelRatio || 1, 2)

    let W = 0, H = 0
    type Node = { x:number; y:number; vx:number; vy:number; radius:number; baseAlpha:number; phase:number }
    type Particle = { x:number; y:number; vx:number; vy:number; radius:number; alpha:number; phase:number; pulseSpeed:number }
    type Pulse = { a:number; b:number; t:number; speed:number; alpha:number }

    let nodes: Node[] = [], particles: Particle[] = [], pulses: Pulse[] = []
    let lastPulseTime = 0
    let mouseX = 0.5, mouseY = 0.5
    let mouseTargetX = 0.5, mouseTargetY = 0.5
    let mousePxX = -1000, mousePxY = -1000
    let animId: number | null = null
    let alive = true

    function initEntities() {
      const isMobile = W < 768
      const nodeCount = isMobile ? 25 : 50
      const particleCount = isMobile ? 35 : 70

      nodes = []
      for (let i = 0; i < nodeCount; i++) {
        nodes.push({
          x: Math.random() * W, y: Math.random() * H,
          vx: (Math.random() - 0.5) * 0.4, vy: (Math.random() - 0.5) * 0.4,
          radius: 1.5 + Math.random() * 1.5,
          baseAlpha: 0.10 + Math.random() * 0.15,
          phase: Math.random() * Math.PI * 2,
        })
      }

      particles = []
      for (let i = 0; i < particleCount; i++) {
        particles.push({
          x: Math.random() * W, y: Math.random() * H,
          vx: (Math.random() - 0.5) * 0.2, vy: (Math.random() - 0.5) * 0.2,
          radius: 0.6 + Math.random() * 0.9,
          alpha: 0.05 + Math.random() * 0.07,
          phase: Math.random() * Math.PI * 2,
          pulseSpeed: 0.0015 + Math.random() * 0.0015,
        })
      }

      pulses = []
    }

    function handleResize() {
      const newW = window.innerWidth, newH = window.innerHeight
      if (newW === W && newH === H) return
      W = newW; H = newH
      canvas.width = W * dpr
      canvas.height = H * dpr
      canvas.style.width = W + 'px'
      canvas.style.height = H + 'px'
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      initEntities()
    }

    function onMouseMove(e: MouseEvent) {
      mouseTargetX = e.clientX / (W || 1)
      mouseTargetY = e.clientY / (H || 1)
      mousePxX = e.clientX
      mousePxY = e.clientY
    }

    function drawBlobs(t: number) {
      mouseX += (mouseTargetX - mouseX) * 0.05
      mouseY += (mouseTargetY - mouseY) * 0.05

      for (const blob of BLOB_CONFIGS) {
        const bx = blob.cx * W + Math.sin(t * blob.speed * Math.PI * 2 + blob.phase) * blob.orbitX * W
        const by = blob.cy * H + Math.cos(t * blob.speed * Math.PI * 2 + blob.phase * 1.3) * blob.orbitY * H
        const px = bx + (mouseX - 0.5) * W * 0.03
        const py = by + (mouseY - 0.5) * H * 0.03
        const r = blob.radius * W
        const grad = ctx.createRadialGradient(px, py, 0, px, py, r)
        const [cr, cg, cb] = blob.color
        grad.addColorStop(0,   `rgba(${cr},${cg},${cb},${blob.alpha})`)
        grad.addColorStop(0.4, `rgba(${cr},${cg},${cb},${blob.alpha * 0.5})`)
        grad.addColorStop(1,   `rgba(${cr},${cg},${cb},0)`)
        ctx.fillStyle = grad
        ctx.beginPath()
        ctx.arc(px, py, r, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    function drawConstellation(t: number) {
      const maxDist = EDGE_MAX_DIST_FRAC * W

      for (const node of nodes) {
        const dx = node.x - mousePxX, dy = node.y - mousePxY
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < MOUSE_REPEL_RADIUS && dist > 0) {
          const force = (1 - dist / MOUSE_REPEL_RADIUS) * MOUSE_REPEL_FORCE
          node.x += (dx / dist) * force
          node.y += (dy / dist) * force
        }
        node.x += node.vx; node.y += node.vy
        if (node.x < 0) { node.x = 0; node.vx *= -1 }
        if (node.x > W) { node.x = W; node.vx *= -1 }
        if (node.y < 0) { node.y = 0; node.vy *= -1 }
        if (node.y > H) { node.y = H; node.vy *= -1 }
      }

      const activeEdges: [number, number][] = []
      ctx.lineWidth = 0.7
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < maxDist) {
            ctx.strokeStyle = `rgba(255,255,255,${0.06 * (1 - dist / maxDist)})`
            ctx.beginPath()
            ctx.moveTo(nodes[i].x, nodes[i].y)
            ctx.lineTo(nodes[j].x, nodes[j].y)
            ctx.stroke()
            activeEdges.push([i, j])
          }
        }
      }

      for (const node of nodes) {
        const alpha = node.baseAlpha * (0.6 + 0.4 * Math.sin(t * 0.002 + node.phase))
        const glowGrad = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.radius * 3)
        glowGrad.addColorStop(0, `rgba(${CONSTELLATION_RGB},${alpha * 0.25})`)
        glowGrad.addColorStop(1, `rgba(${CONSTELLATION_RGB},0)`)
        ctx.fillStyle = glowGrad
        ctx.beginPath(); ctx.arc(node.x, node.y, node.radius * 3, 0, Math.PI * 2); ctx.fill()
        ctx.fillStyle = `rgba(${CONSTELLATION_RGB},${alpha})`
        ctx.beginPath(); ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2); ctx.fill()
      }

      if (activeEdges.length > 0 && t - lastPulseTime > PULSE_INTERVAL && pulses.length < MAX_PULSES) {
        const [a, b] = activeEdges[Math.floor(Math.random() * activeEdges.length)]
        pulses.push({ a, b, t: 0, speed: 0.01 + Math.random() * 0.015, alpha: 0.20 + Math.random() * 0.15 })
        lastPulseTime = t
      }

      for (let i = pulses.length - 1; i >= 0; i--) {
        const pulse = pulses[i]
        pulse.t += pulse.speed
        if (pulse.t > 1) { pulses.splice(i, 1); continue }
        const nA = nodes[pulse.a], nB = nodes[pulse.b]
        if (!nA || !nB) { pulses.splice(i, 1); continue }
        const ppx = nA.x + (nB.x - nA.x) * pulse.t
        const ppy = nA.y + (nB.y - nA.y) * pulse.t
        let fade = 1
        if (pulse.t < 0.2) fade = pulse.t / 0.2
        else if (pulse.t > 0.8) fade = (1 - pulse.t) / 0.2
        const a = pulse.alpha * fade
        const grad = ctx.createRadialGradient(ppx, ppy, 0, ppx, ppy, 5)
        grad.addColorStop(0,   `rgba(${CONSTELLATION_RGB},${a})`)
        grad.addColorStop(0.5, `rgba(${CONSTELLATION_RGB},${a * 0.15})`)
        grad.addColorStop(1,   `rgba(${CONSTELLATION_RGB},0)`)
        ctx.fillStyle = grad
        ctx.beginPath(); ctx.arc(ppx, ppy, 5, 0, Math.PI * 2); ctx.fill()
      }
    }

    function drawParticles(t: number) {
      for (const p of particles) {
        p.x += p.vx; p.y += p.vy
        if (p.x < 0) p.x = W; if (p.x > W) p.x = 0
        if (p.y < 0) p.y = H; if (p.y > H) p.y = 0
        const alpha = p.alpha * (0.5 + 0.5 * Math.sin(t * p.pulseSpeed + p.phase))
        ctx.fillStyle = `rgba(255,255,255,${alpha})`
        ctx.beginPath(); ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2); ctx.fill()
      }
    }

    function animate(timestamp: number) {
      if (!alive) return
      if (!W || !H) handleResize()
      ctx.fillStyle = CANVAS_BG
      ctx.fillRect(0, 0, W, H)
      drawBlobs(timestamp)
      drawConstellation(timestamp)
      drawParticles(timestamp)
      animId = requestAnimationFrame(animate)
    }

    handleResize()
    window.addEventListener('resize', handleResize)
    window.addEventListener('mousemove', onMouseMove, { passive: true })
    animId = requestAnimationFrame(animate)

    return () => {
      alive = false
      if (animId) cancelAnimationFrame(animId)
      window.removeEventListener('resize', handleResize)
      window.removeEventListener('mousemove', onMouseMove)
    }
  }, [])

  return <canvas ref={canvasRef} className={className} />
}
