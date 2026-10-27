// Lesson 14 · an impedance-controlled hand pressed into a rigid surface, reduced to one axis.
// The hand has the inertia the lesson's Panda feels along z (Λzz = 6.88 kg); the controller is
// F = K(x_target − x) − D ẋ with D = 2ζ√(KΛ); the target ramps (quintic, 1 s) from 3 cm above the
// surface to a chosen depth below it and holds, as in part B. The surface is a toy soft contact
// that stiffens sharply with penetration plus a damper, standing in for MuJoCo's soft contact.
// Simulated with semi-implicit Euler at 0.1 ms whenever a slider changes, then played back.
import { animate, axes, colors, frame, mono, type Lang } from './_kit'

const LAMBDA = 6.88, START = 0.03, T_RAMP = 1, T_END = 2.5, H = 1e-4
// toy contact: F = k0 (exp(p/p0) − 1) + b·(penetration speed), never pulling
const P0 = 1e-4, K0 = 5 / (Math.exp(5.45) - 1), B = 800

function simulate(K: number, zeta: number, depth: number) {
	const D = 2 * zeta * Math.sqrt(K * LAMBDA)
	let x = START, v = 0
	const ts: number[] = [], xs: number[] = [], xt: number[] = [], fs: number[] = []
	let peak = 0, holdSum = 0, holdN = 0, pen = 0
	const n = Math.round(T_END / H)
	for (let i = 0; i <= n; i++) {
		const t = i * H
		const u = Math.min(t / T_RAMP, 1)
		const target = START - (START + depth) * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)
		const F = K * (target - x) - D * v
		const p = -x
		let Fc = 0
		if (p > 0) Fc = Math.max(0, K0 * (Math.exp(Math.min(p / P0, 60)) - 1) - B * v)
		v += (H * (F + Fc)) / LAMBDA
		x += H * v
		peak = Math.max(peak, Fc)
		if (t > T_END - 0.2) { holdSum += Fc; holdN++; pen = Math.max(pen, p) }
		if (i % 10 === 0) { ts.push(t); xs.push(x); xt.push(target); fs.push(Fc) }
	}
	return { ts, xs, xt, fs, peak, hold: holdSum / holdN, pen }
}

export default function (root: HTMLElement, lang: Lang) {
	const T_ = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => (lang === 'tr' ? x.toFixed(d).replace('.', ',') : x.toFixed(d))
	const f = frame(root, lang, {
		title: { en: 'A spring you design, pressed into a table', tr: 'Tasarladığın bir yay, masaya bastırılıyor' },
		caption: {
			en: 'The target (dashed ring) moves from 3 cm above the surface to a depth below it; the hand is tied to it by a virtual spring K and damper D. Once the hand rests on the surface it pushes with about K × depth, whatever the contact is like. The first spike is the impact, set by how fast the hand arrives and how the contact stops it. The 1-D toy cannot show what happens past the joint torque limits (Break it).',
			tr: 'Hedef (kesikli halka) yüzeyin 3 cm üstünden altındaki bir derinliğe iniyor; el ona sanal bir yay K ve damper D ile bağlı. El yüzeye oturunca, contact nasıl olursa olsun yaklaşık K × derinlik ile itiyor. İlk sivri tepe çarpma: elin ne hızla geldiği ve contact’ın onu nasıl durdurduğu belirliyor. Tek boyutlu oyuncak model, joint torque limitlerinin ötesinde olanı gösteremez (Boz).',
		},
	})
	const { ctx, W, H: CH } = f.canvas(640, 300)
	const logK = f.slider({ en: 'stiffness K [N/m]', tr: 'stiffness K [N/m]' }, { min: 2, max: Math.log10(3000), step: 0.01, value: 3, fmt: (x) => String(Math.round(10 ** x)) })
	const zeta = f.slider({ en: 'damping ratio ζ', tr: 'damping oranı ζ' }, { min: 0.2, max: 1.5, step: 0.05, value: 1, fmt: (x) => num(x, 2) })
	const depth = f.slider({ en: 'target depth [cm]', tr: 'hedef derinliği [cm]' }, { min: 1, max: 5, step: 0.5, value: 5, fmt: (x) => num(x, 1) })
	const rDesign = f.readout({ en: 'K × depth', tr: 'K × derinlik' })
	const rHold = f.readout({ en: 'holding force', tr: 'tutma kuvveti' })
	const rPeak = f.readout({ en: 'impact peak', tr: 'çarpma tepesi' })
	const rPen = f.readout({ en: 'penetration', tr: 'penetration' })

	let sim = simulate(10 ** logK.get(), zeta.get(), depth.get() / 100)
	let clock = 0
	const rerun = () => { sim = simulate(10 ** logK.get(), zeta.get(), depth.get() / 100); clock = 0 }
	for (const c of [logK, zeta, depth]) c.on(rerun)

	animate(root, (dt) => {
		const c = colors()
		clock = (clock + dt) % (T_END + 0.8)
		const t = Math.min(clock, T_END)
		const i = Math.min(sim.ts.length - 1, Math.round(t / (10 * H)))
		ctx.clearRect(0, 0, W, CH)
		ctx.font = mono

		// scene: the surface at y = SY, 1 cm = 20 px; the probe's lowest point is the hand position
		const SX = 105, SY = 160, PX = 2000, R = 12
		ctx.fillStyle = c.soft; ctx.fillRect(20, SY, 190, CH - SY - 6)
		ctx.strokeStyle = c.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(20, SY); ctx.lineTo(210, SY); ctx.stroke()
		ctx.fillStyle = c.muted; ctx.fillText(T_('surface', 'yüzey'), 24, SY + 20)
		const yHand = SY - sim.xs[i] * PX, yTarget = SY - sim.xt[i] * PX
		// virtual spring between the hand and its target (the target may sit inside the surface)
		const y0 = yHand - R, y1 = yTarget, coils = 8
		ctx.strokeStyle = c.muted; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(SX, y0)
		for (let k = 1; k < coils * 2; k++) ctx.lineTo(SX + (k % 2 ? 8 : -8), y0 + ((y1 - y0) * k) / (coils * 2))
		ctx.lineTo(SX, y1); ctx.stroke()
		ctx.setLineDash([4, 4]); ctx.strokeStyle = c.ink; ctx.lineWidth = 1.5
		ctx.beginPath(); ctx.arc(SX, yTarget, 8, 0, 7); ctx.stroke(); ctx.setLineDash([])
		ctx.fillStyle = c.muted; ctx.fillText(T_('target', 'hedef'), SX - 70, yTarget + 4)
		ctx.fillStyle = c.accent; ctx.beginPath(); ctx.arc(SX, yHand - R, R, 0, 7); ctx.fill()
		ctx.fillStyle = c.muted; ctx.fillText(T_('hand', 'el'), SX - 50, yHand - R + 4)
		// contact force arrow on the surface
		const Fnow = sim.fs[i]
		if (Fnow > 0.05) {
			const len = Math.min(60, 10 + 18 * Math.log10(1 + Fnow))
			ctx.strokeStyle = c.accent; ctx.fillStyle = c.accent; ctx.lineWidth = 3
			ctx.beginPath(); ctx.moveTo(SX + 40, SY + 4); ctx.lineTo(SX + 40, SY + 4 + len); ctx.stroke()
			ctx.beginPath(); ctx.moveTo(SX + 34, SY + len - 2); ctx.lineTo(SX + 46, SY + len - 2); ctx.lineTo(SX + 40, SY + len + 8); ctx.fill()
			ctx.fillText(num(Fnow, 1) + ' N', SX + 52, SY + 24)
		}

		// force on the surface over time
		const fMax = Math.max(sim.peak, 10 ** logK.get() * depth.get() / 100) * 1.15
		const a = axes(ctx, { x0: 260, y0: 18, x1: W - 14, y1: CH - 30 }, [0, T_END], [0, fMax])
		a.draw(c, T_('time [s] →', 'zaman [s] →'), T_('force on surface [N]', 'yüzeydeki kuvvet [N]'))
		const design = (10 ** logK.get() * depth.get()) / 100
		a.line([[0, design], [T_END, design]], c.muted, 1.5, [5, 5])
		ctx.fillStyle = c.muted
		const lbl = T_('K × depth', 'K × derinlik')
		ctx.fillText(lbl, W - 14 - ctx.measureText(lbl).width, a.Y(design) - 6)
		const pts: [number, number][] = []
		for (let j = 0; j <= i; j++) pts.push([sim.ts[j], sim.fs[j]])
		a.line(pts, c.accent, 2.5)

		rDesign(num(design, 1) + ' N')
		rHold(num(sim.hold, 2) + ' N')
		rPeak(num(sim.peak, 1) + ' N')
		rPen(num(sim.pen * 1000, 2) + ' mm')
	})
}
