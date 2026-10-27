// Lesson 13 · time scalings for one joint move of Δq = 0.6 rad (the lesson's largest joint).
// Linear, trapezoid (accelerate T/3, cruise, decelerate T/3), cubic and quintic, drawn as position,
// velocity and acceleration over time for a chosen duration T. Pure formulas, no simulation; the
// dashed band is the 10 rad/s² joint-acceleration limit quoted from libfranka in the lesson.
import { animate, axes, colors, frame, mono, type Lang } from './_kit'

type Shape = { s: (u: number) => number; ds: (u: number) => number; dds: (u: number) => number }

const SHAPES: Shape[] = [
	{ s: (u) => u, ds: () => 1, dds: () => 0 },
	{
		s: (u) => (u < 1 / 3 ? 2.25 * u * u : u < 2 / 3 ? 0.25 + 1.5 * (u - 1 / 3) : 1 - 2.25 * (1 - u) ** 2),
		ds: (u) => (u < 1 / 3 ? 4.5 * u : u < 2 / 3 ? 1.5 : 4.5 * (1 - u)),
		dds: (u) => (u < 1 / 3 ? 4.5 : u < 2 / 3 ? 0 : -4.5),
	},
	{ s: (u) => 3 * u * u - 2 * u ** 3, ds: (u) => 6 * u - 6 * u * u, dds: (u) => 6 - 12 * u },
	{
		s: (u) => 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5,
		ds: (u) => 30 * u * u - 60 * u ** 3 + 30 * u ** 4,
		dds: (u) => 60 * u - 180 * u * u + 120 * u ** 3,
	},
]
// peak |s'| and |s''| of each shape on [0, 1] (linear: velocity 1, acceleration unbounded)
const PEAK_DS = [1, 1.5, 1.5, 1.875]
const PEAK_DDS = [Infinity, 4.5, 6, (10 * Math.sqrt(3)) / 3]
// where each shape's acceleration jumps (as a fraction of T)
const JUMPS = [[0, 1], [0, 1 / 3, 2 / 3, 1], [0, 1], []]

export default function (root: HTMLElement, lang: Lang) {
	const T_ = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => (lang === 'tr' ? x.toFixed(d).replace('.', ',') : x.toFixed(d))
	const f = frame(root, lang, {
		title: { en: 'Same move, four schedules', tr: 'Aynı hareket, dört zamanlama' },
		caption: {
			en: 'One joint moves 0.6 rad in T seconds. Pick a shape and a duration: the faint curves are the other shapes. Circles mark where the acceleration jumps, which means unbounded jerk. Only the quintic starts and ends with zero velocity and zero acceleration. Halve T and watch the acceleration grow fourfold. Dashed: the 10 rad/s² joint-acceleration limit the lesson quotes for the Panda.',
			tr: 'Bir eklem T saniyede 0,6 rad dönüyor. Bir şekil ve bir süre seç: soluk eğriler diğer şekiller. Halkalar ivmenin sıçradığı yerleri gösterir; orada jerk sınırsızdır. Sıfır hız ve sıfır ivmeyle başlayıp biten yalnızca quintic. T’yi yarıya indir, ivmenin dört kat büyüdüğünü izle. Kesikli: dersin Panda için aktardığı 10 rad/s²’lik joint ivme sınırı.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 380)
	const names = [
		{ en: 'linear', tr: 'linear' }, { en: 'trapezoid', tr: 'trapezoid' },
		{ en: 'cubic', tr: 'cubic' }, { en: 'quintic (minimum jerk)', tr: 'quintic (minimum jerk)' },
	]
	const shape = f.select({ en: 'shape', tr: 'şekil' }, names, 3)
	const dur = f.slider({ en: 'duration T [s]', tr: 'süre T [s]' }, { min: 0.25, max: 2, step: 0.05, value: 1, fmt: (x) => num(x, 2) })
	const rVel = f.readout({ en: 'peak velocity', tr: 'tepe hız' })
	const rAcc = f.readout({ en: 'peak acceleration', tr: 'tepe ivme' })
	const rJerk = f.readout({ en: 'peak jerk', tr: 'tepe jerk' })
	const rLim = f.readout({ en: 'within 10 rad/s²', tr: '10 rad/s² içinde' })

	const DQ = 0.6, TMIN = -0.15, TMAX = 2.15, PAUSE = 0.5
	let clock = 0

	animate(root, (dt) => {
		const c = colors()
		const k = shape.get(), T = dur.get()
		clock = (clock + dt) % (T + 2 * PAUSE)
		const tNow = clock - PAUSE
		const uNow = Math.min(1, Math.max(0, tNow / T))

		const vMax = (1.875 * DQ) / T * 1.15
		const aMax = (6 * DQ) / (T * T) * 1.15
		const curve = (i: number, kind: 0 | 1 | 2): [number, number][] => {
			const sh = SHAPES[i], pts: [number, number][] = []
			const end = kind === 0 ? DQ : 0
			pts.push([TMIN, 0], [0, 0])
			for (let j = 0; j <= 300; j++) {
				const u = j / 300
				const v = kind === 0 ? DQ * sh.s(u) : kind === 1 ? (DQ * sh.ds(u)) / T : (DQ * sh.dds(u)) / (T * T)
				pts.push([u * T, v])
			}
			pts.push([T, end], [TMAX, end])
			return pts
		}

		ctx.clearRect(0, 0, W, H)
		ctx.font = mono
		// the joint itself, sweeping 0.6 rad
		const px = 92, py = 215, L = 74
		const ang = (q: number) => -0.3 + q
		const tip = (q: number): [number, number] => [px + L * Math.cos(ang(q)), py - L * Math.sin(ang(q))]
		ctx.strokeStyle = c.line; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4])
		for (const q of [0, DQ]) { const [x, y] = tip(q); ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(x, y); ctx.stroke() }
		ctx.setLineDash([])
		const q = DQ * SHAPES[k].s(uNow)
		const [tx, ty] = tip(q)
		ctx.strokeStyle = c.ink; ctx.lineWidth = 9; ctx.lineCap = 'round'
		ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(tx, ty); ctx.stroke(); ctx.lineCap = 'butt'
		ctx.fillStyle = c.accent; ctx.beginPath(); ctx.arc(tx, ty, 9, 0, 7); ctx.fill()
		ctx.fillStyle = c.bg; ctx.strokeStyle = c.ink; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(px, py, 7, 0, 7); ctx.fill(); ctx.stroke()
		ctx.fillStyle = c.muted
		ctx.fillText(`Δq = ${num(DQ, 1)} rad`, 30, 120)
		ctx.fillText(`t = ${num(Math.min(Math.max(tNow, 0), T), 2)} s`, 30, 320)

		// three stacked panels: position, velocity, acceleration
		const x0 = 250, x1 = W - 14
		const panels: { y0: number; y1: number; yr: [number, number]; kind: 0 | 1 | 2; label: string }[] = [
			{ y0: 14, y1: 104, yr: [-0.05, 0.68], kind: 0, label: T_('q [rad]', 'q [rad]') },
			{ y0: 128, y1: 218, yr: [-0.1 * vMax, vMax], kind: 1, label: T_('q̇ [rad/s]', 'q̇ [rad/s]') },
			{ y0: 242, y1: 350, yr: [-aMax, aMax], kind: 2, label: T_('q̈ [rad/s²]', 'q̈ [rad/s²]') },
		]
		for (const p of panels) {
			const a = axes(ctx, { x0, y0: p.y0, x1, y1: p.y1 }, [TMIN, TMAX], p.yr)
			a.draw(c, p.kind === 2 ? T_('time [s] →', 'zaman [s] →') : '', p.label)
			a.line([[TMIN, 0], [TMAX, 0]], c.line, 1)
			if (p.kind === 2 && 10 < aMax) {
				a.line([[TMIN, 10], [TMAX, 10]], c.muted, 1.2, [5, 5])
				a.line([[TMIN, -10], [TMAX, -10]], c.muted, 1.2, [5, 5])
				ctx.fillStyle = c.muted
				ctx.fillText('±10', x1 - ctx.measureText('±10').width, a.Y(10) - 4)
			}
			for (let i = 0; i < 4; i++) if (i !== k) a.line(curve(i, p.kind), c.line, 1.5)
			if (p.kind === 2 && k === 0) {
				// linear: zero acceleration with two unbounded spikes at the ends
				a.line([[TMIN, 0], [TMAX, 0]], c.accent, 2.5)
				ctx.strokeStyle = c.accent; ctx.lineWidth = 2.5
				for (const [t, sg] of [[0, 1], [T, -1]]) {
					const X = a.X(t)
					ctx.beginPath(); ctx.moveTo(X, a.Y(0)); ctx.lineTo(X, a.Y(sg * aMax * 0.95)); ctx.stroke()
					ctx.fillStyle = c.accent; ctx.fillText('∞', X + 5, a.Y(sg * aMax * 0.8) + 5)
				}
			} else {
				a.line(curve(k, p.kind), c.accent, 2.5)
			}
			// jumps of the selected shape: in acceleration, and for linear also in velocity
			const jumps = p.kind === 2 ? JUMPS[k] : p.kind === 1 && k === 0 ? [0, 1] : []
			ctx.strokeStyle = c.ink; ctx.lineWidth = 1.5
			const g = (u: number) => (p.kind === 1 ? (DQ * SHAPES[k].ds(u)) / T : (DQ * SHAPES[k].dds(u)) / (T * T))
			for (const u of jumps) {
				// circle the middle of the vertical jump (rest is zero outside [0, T])
				const before = u <= 0 ? 0 : g(u - 1e-6), after = u >= 1 ? 0 : g(u + 1e-6)
				const mid = k === 0 && p.kind === 2 ? 0 : (before + after) / 2
				ctx.beginPath(); ctx.arc(a.X(u * T), a.Y(mid), 7, 0, 7); ctx.stroke()
			}
			// time cursor
			const tc = Math.min(Math.max(tNow, TMIN), TMAX)
			ctx.strokeStyle = c.muted; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a.X(tc), p.y0); ctx.lineTo(a.X(tc), p.y1); ctx.stroke()
		}

		const vPeak = (PEAK_DS[k] * DQ) / T, aPeak = (PEAK_DDS[k] * DQ) / (T * T)
		rVel(num(vPeak, 2) + ' rad/s')
		rAcc(Number.isFinite(aPeak) ? num(aPeak, 2) + ' rad/s²' : '∞')
		rJerk(k === 3 ? num((60 * DQ) / T ** 3, 1) + ' rad/s³' : T_('∞ (acceleration jumps)', '∞ (ivme sıçrıyor)'))
		rLim(aPeak <= 10 ? T_('yes', 'evet') : T_('no', 'hayır'))
	})
}
