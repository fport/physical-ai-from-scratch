// Lesson 04 · a cube on a ramp with Coulomb friction (one μ, as in MuJoCo). The contact can push
// back with any force inside the friction cone (half-angle arctan μ around the surface normal).
// Holding still needs a reaction straight up (it must cancel the weight); when that direction leaves
// the cone, the cube slides with a = g(sin θ − μ cos θ). The soft-contact creep is not modelled.
import { animate, colors, frame, mono, type Lang } from './_kit'

const MASSES = [0.1, 1, 10] // kg, the lesson's three cubes
const g = 9.81

export default function (root: HTMLElement, lang: Lang) {
	const L = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => x.toFixed(d).replace('.', L('.', ','))
	const f = frame(root, lang, {
		title: { en: 'Inside or outside the cone', tr: 'Cone’un içinde mi dışında mı' },
		caption: {
			en: 'To stay put, the ramp must push the cube straight up, against its weight. Friction allows any push inside the cone around the surface normal, with half-angle arctan μ. Tilt the ramp until “straight up” leaves the cone. Then change the mass: both forces grow, the angle does not move.',
			tr: 'Küpün yerinde kalması için rampanın onu ağırlığına karşı dümdüz yukarı itmesi gerekir. Friction, yüzey normali etrafında yarı açısı arctan μ olan cone’un içindeki her itişe izin verir. “Dümdüz yukarı” cone’dan çıkana kadar rampayı eğ. Sonra kütleyi değiştir: iki kuvvet de büyür, açı yerinden oynamaz.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 300)
	const th = f.slider({ en: 'ramp angle θ', tr: 'rampa açısı θ' }, { min: 0, max: 60, step: 0.5, value: 20, fmt: (x) => `${num(x, 1)}°` })
	const mu = f.slider({ en: 'friction μ', tr: 'friction μ' }, { min: 0.1, max: 1.25, step: 0.05, value: 0.5, fmt: (x) => num(x, 2) })
	const ms = f.slider({ en: 'mass', tr: 'kütle' }, { min: 0, max: 2, step: 1, value: 1, fmt: (i) => `${num(MASSES[i], 1)} kg` })
	const rTan = f.readout({ en: 'tan θ', tr: 'tan θ' })
	const rSlip = f.readout({ en: 'slip angle arctan μ', tr: 'kayma açısı arctan μ' })
	const rPull = f.readout({ en: 'pull m·g·sin θ', tr: 'çekiş m·g·sin θ' })
	const rBud = f.readout({ en: 'budget μ·m·g·cos θ', tr: 'bütçe μ·m·g·cos θ' })
	const rState = f.readout({ en: 'cube', tr: 'küp' })

	const P0 = { x: 120, y: 272 }, LR = 300, BW = 46, S0 = 210, PX_PER_M = 300
	let s = S0, v = 0
	const reset = () => { s = S0; v = 0 }
	for (const c of [th, mu, ms]) c.on(reset)
	f.button({ en: 'Reset cube', tr: 'Küpü geri koy' }, reset)

	const arrow = (x: number, y: number, dx: number, dy: number, color: string, width = 2.5, dash: number[] = []) => {
		const len = Math.hypot(dx, dy)
		if (len < 1) return
		ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash)
		ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + dx, y + dy); ctx.stroke(); ctx.setLineDash([])
		const ux = dx / len, uy = dy / len
		ctx.beginPath(); ctx.moveTo(x + dx, y + dy)
		ctx.lineTo(x + dx - 9 * ux - 4.5 * uy, y + dy - 9 * uy + 4.5 * ux)
		ctx.lineTo(x + dx - 9 * ux + 4.5 * uy, y + dy - 9 * uy - 4.5 * ux)
		ctx.fill()
	}

	animate(root, (dt) => {
		const theta = (th.get() * Math.PI) / 180, m = MASSES[ms.get()], alpha = Math.atan(mu.get())
		const slides = Math.tan(theta) > mu.get()
		const acc = slides ? g * (Math.sin(theta) - mu.get() * Math.cos(theta)) : 0
		if (slides) {
			v += acc * dt * 0.5
			s -= v * dt * 0.5 * PX_PER_M
			if (s < BW / 2 + 4) reset()
		}
		const c = colors()
		ctx.clearRect(0, 0, W, H)
		ctx.font = mono
		const u = { x: Math.cos(theta), y: -Math.sin(theta) } // up the ramp (canvas y points down)
		const n = { x: -Math.sin(theta), y: -Math.cos(theta) } // surface normal, away from the ramp
		const P1 = { x: P0.x + LR * u.x, y: P0.y + LR * u.y }
		// ramp body
		ctx.fillStyle = c.soft
		ctx.beginPath(); ctx.moveTo(P0.x, P0.y); ctx.lineTo(P1.x, P1.y); ctx.lineTo(P1.x, P0.y); ctx.closePath(); ctx.fill()
		ctx.strokeStyle = c.ink; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(P0.x, P0.y); ctx.lineTo(P1.x, P1.y); ctx.stroke()
		ctx.strokeStyle = c.line; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(20, P0.y); ctx.lineTo(W - 20, P0.y); ctx.stroke()
		// cube
		const C = { x: P0.x + s * u.x, y: P0.y + s * u.y }
		const corner = (a: number, b: number) => ({ x: C.x + a * u.x + b * n.x, y: C.y + a * u.y + b * n.y })
		const q = [corner(-BW / 2, 0), corner(BW / 2, 0), corner(BW / 2, BW), corner(-BW / 2, BW)]
		ctx.fillStyle = c.ink; ctx.beginPath(); q.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))); ctx.closePath(); ctx.fill()
		// friction cone at the contact
		const CL = 120
		const rot = (vx: number, vy: number, b: number) => ({ x: vx * Math.cos(b) - vy * Math.sin(b), y: vx * Math.sin(b) + vy * Math.cos(b) })
		const e1 = rot(n.x, n.y, alpha), e2 = rot(n.x, n.y, -alpha)
		ctx.fillStyle = c.soft; ctx.globalAlpha = 0.9
		ctx.beginPath(); ctx.moveTo(C.x, C.y); ctx.lineTo(C.x + CL * e1.x, C.y + CL * e1.y); ctx.lineTo(C.x + CL * e2.x, C.y + CL * e2.y); ctx.closePath(); ctx.fill()
		ctx.globalAlpha = 1
		ctx.strokeStyle = c.accent; ctx.lineWidth = 1.5
		ctx.beginPath(); ctx.moveTo(C.x + CL * e1.x, C.y + CL * e1.y); ctx.lineTo(C.x, C.y); ctx.lineTo(C.x + CL * e2.x, C.y + CL * e2.y); ctx.stroke()
		ctx.strokeStyle = c.muted; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(C.x, C.y); ctx.lineTo(C.x + CL * n.x, C.y + CL * n.y); ctx.stroke(); ctx.setLineDash([])
		// forces: weight and the reaction needed to hold (straight up), lengths grow with mass
		const F = 34 * Math.pow(m, 0.25) // px for m·g, compressed so 0.1 kg and 10 kg both fit
		const G = { x: C.x + (BW / 2) * n.x, y: C.y + (BW / 2) * n.y }
		arrow(G.x, G.y, 0, F * 1.6, c.ink, 2.5)
		arrow(G.x, G.y, -u.x * F * 1.6 * Math.sin(theta), -u.y * F * 1.6 * Math.sin(theta), c.muted, 1.5, [4, 3])
		if (!slides) arrow(C.x, C.y, 0, -F * 1.6, c.accent, 3)
		else {
			arrow(C.x, C.y, 0, -F * 1.6, c.muted, 2, [5, 4])
			const Nlen = F * 1.6 * Math.cos(theta) / Math.cos(alpha)
			const edge = e1.x * u.x + e1.y * u.y > e2.x * u.x + e2.y * u.y ? e1 : e2 // friction acts uphill: the edge leaning up the ramp
			arrow(C.x, C.y, edge.x * Nlen, edge.y * Nlen, c.accent, 3)
		}
		// labels
		ctx.fillStyle = c.muted
		ctx.fillText(L('cone: ±arctan μ around the normal', 'cone: normal etrafında ±arctan μ'), 330, 24)
		ctx.fillText(slides ? L('needed (dashed) is outside the cone', 'gereken (kesikli) cone’un dışında') : L('thick arrow: push needed to hold, inside', 'kalın ok: tutmak için gereken itiş, içeride'), 330, 44)
		ctx.fillText('↓ m·g', G.x + 8, G.y + F * 1.6)
		ctx.fillText(`θ = ${num(th.get(), 1)}°`, P0.x + 40, P0.y - 8)

		rTan(num(Math.tan(theta), 3))
		rSlip(`${num((alpha * 180) / Math.PI, 2)}°`)
		rPull(`${num(m * g * Math.sin(theta), 2)} N`)
		rBud(`${num(mu.get() * m * g * Math.cos(theta), 2)} N`)
		rState(slides ? L(`slides, a = ${num(acc, 2)} m/s²`, `kayıyor, a = ${num(acc, 2)} m/s²`) : L('holds', 'duruyor'))
	})
}
