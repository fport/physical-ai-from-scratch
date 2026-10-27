// Lesson 39 · how wide to randomize. A toy version of the lesson's experiment: one joint with the
// lesson's link (0.5 kg at 15 cm, armature 0.028, damping 0.6, friction 0.052, torque limit 2.94 N·m,
// gravity left out) tracks the lesson's 3 s reference through a measurement delay. The "policy" is
// a single gain Kp (Kd tied to it by ζ = 0.7). Only the delay is randomized: the training delays
// are spread evenly over [5, 5 + 40w] ms, and Kp is picked from a grid to minimize the mean RMS
// tracking error over them. The error table over (Kp, delay) is computed once in the browser.
import { animate, axes, colors, frame, mono, type Lang } from './_kit'

export default function (root: HTMLElement, lang: Lang) {
	const T = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => (lang === 'tr' ? x.toFixed(d).replace('.', ',') : x.toFixed(d))
	const f = frame(root, lang, {
		title: { en: 'Randomize the delay: how wide?', tr: 'Gecikmeyi randomize et: ne kadar geniş?' },
		caption: {
			en: 'Widen the training range and the tuned gain drops. Too narrow and the real delay lies outside it, where the stiff loop fails; too wide and the soft loop tracks badly everywhere. The lesson’s held-out plants had delays of 15.5–28.8 ms. Success line: 2° RMS, as in the lesson.',
			tr: 'Training aralığını genişlet; ayarlanan gain düşer. Çok dar olursa gerçek gecikme aralığın dışında kalır ve sert döngü orada başarısız olur; çok geniş olursa yumuşak döngü her yerde kötü takip eder. Dersteki held-out plant’lerin gecikmesi 15,5–28,8 ms idi. Başarı çizgisi: dersteki gibi 2° RMS.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 290)
	const width = f.slider({ en: 'randomization width w', tr: 'randomization genişliği w' }, { min: 0, max: 2, step: 0.05, value: 0, fmt: (x) => num(x, 2) })
	const real = f.slider({ en: 'real delay [ms]', tr: 'gerçek gecikme [ms]' }, { min: 0, max: 80, step: 0.5, value: 22.5, fmt: (x) => num(x, 1) })
	const rRange = f.readout({ en: 'training delays', tr: 'training gecikmeleri' })
	const rKp = f.readout({ en: 'tuned Kp', tr: 'ayarlanan Kp' })
	const rNom = f.readout({ en: 'RMS at 5 ms', tr: '5 ms’de RMS' })
	const rReal = f.readout({ en: 'RMS at the real delay', tr: 'gerçek gecikmede RMS' })

	// plant and reference (lesson 39), toy integrator at 2.5 ms
	const I = 0.5 * 0.15 * 0.15 + 0.028, B = 0.6, FRIC = 0.052, TMAX = 2.94, ZETA = 0.7
	const h = 0.0025, N = 1200, CAP = 20
	const R = new Float64Array(N), DR = new Float64Array(N)
	for (let k = 0; k < N; k++) {
		const t = k * h
		R[k] = 0.5 * Math.sin(2 * Math.PI * 0.4 * t) + 0.25 * Math.sin(2 * Math.PI * 1.1 * t + 1)
		DR[k] = 0.5 * 2 * Math.PI * 0.4 * Math.cos(2 * Math.PI * 0.4 * t) + 0.25 * 2 * Math.PI * 1.1 * Math.cos(2 * Math.PI * 1.1 * t + 1)
	}
	const hq = new Float64Array(N), hv = new Float64Array(N)
	const rms = (Kp: number, nd: number) => {
		const Kd = 2 * ZETA * Math.sqrt(Kp * I)
		let q = R[0], v = DR[0], se = 0
		for (let k = 0; k < N; k++) {
			hq[k] = q; hv[k] = v
			const j = Math.max(0, k - nd)
			const u = Math.max(-TMAX, Math.min(TMAX, Kp * (R[k] - hq[j]) - Kd * hv[j]))
			v += (h * (u - B * v - FRIC * Math.tanh(v / 0.01))) / I
			q += h * v
			const e = q - R[k]; se += e * e
			if (Math.abs(e) > 1) return CAP
		}
		return Math.min(CAP, (Math.sqrt(se / N) * 180) / Math.PI)
	}
	const KP: number[] = []
	for (let i = 0; i < 48; i++) KP.push(5 * Math.pow(400, i / 47))
	const ND = 41 // delays 0, 2.5, …, 100 ms
	let J: Float64Array[] | null = null

	const tune = (w: number) => {
		const hi = 5 + 40 * w
		const idx: number[] = []
		for (let n = 0; n < ND; n++) { const d = n * 2.5; if (d >= 5 - 1e-9 && d <= hi + 1e-9) idx.push(n) }
		let best = 0, bv = Infinity
		KP.forEach((_, a) => { const m = idx.reduce((s, n) => s + J![a][n], 0) / idx.length; if (m < bv) { bv = m; best = a } })
		return best
	}
	const at = (a: number, d: number) => {
		const x = Math.max(0, Math.min(ND - 1, d / 2.5)), i = Math.min(ND - 2, Math.floor(x)), u = x - i
		return J![a][i] * (1 - u) + J![a][i + 1] * u
	}

	animate(root, () => {
		if (!J) J = KP.map((kp) => Float64Array.from({ length: ND }, (_, n) => rms(kp, n)))
		const c = colors()
		const w = width.get(), dReal = real.get()
		const a = tune(w), a0 = tune(0)
		ctx.clearRect(0, 0, W, H)
		const YM = 8
		const ax = axes(ctx, { x0: 56, y0: 24, x1: W - 16, y1: H - 34 }, [0, 100], [0, YM])
		// training range
		const hi = Math.min(100, 5 + 40 * w)
		ctx.fillStyle = c.soft
		ctx.fillRect(ax.X(5) - (w === 0 ? 2 : 0), ax.Y(YM), Math.max(4, ax.X(hi) - ax.X(5)), ax.Y(0) - ax.Y(YM))
		ctx.fillStyle = c.muted; ctx.font = mono
		ctx.fillText(T('trained on', 'training'), ax.X(5) + 4, ax.Y(YM) + 14)
		// success threshold
		ax.line([[0, 2], [100, 2]], c.line, 1.5, [5, 5])
		ctx.fillText('2°', ax.X(100) - 22, ax.Y(2) - 6)
		const curve = (k: number): [number, number][] => Array.from({ length: ND }, (_, n) => [n * 2.5, Math.min(YM, J![k][n])])
		ctx.save(); ctx.beginPath(); ctx.rect(56, 24, W - 72, H - 58); ctx.clip()
		if (a0 !== a) ax.line(curve(a0), c.muted, 2, [6, 4])
		ax.line(curve(a), c.accent)
		ctx.restore()
		// real plant
		const eReal = at(a, dReal)
		ctx.strokeStyle = c.ink; ctx.lineWidth = 1.5
		ctx.beginPath(); ctx.moveTo(ax.X(dReal), ax.Y(YM)); ctx.lineTo(ax.X(dReal), ax.Y(0)); ctx.stroke()
		ctx.fillStyle = c.ink; ctx.beginPath(); ctx.arc(ax.X(dReal), ax.Y(Math.min(YM, eReal)), 5, 0, 7); ctx.fill()
		const lab = T('real', 'gerçek')
		ctx.fillText(lab, Math.min(ax.X(dReal) + 6, W - 16 - ctx.measureText(lab).width), ax.Y(YM) + 30)
		ax.draw(c, T('plant delay [ms] →', 'plant gecikmesi [ms] →'), T('RMS tracking error [°]', 'RMS takip hatası [°]'))
		ctx.fillStyle = c.muted
		for (const s of [0, 20, 40, 60, 80]) ctx.fillText(String(s), ax.X(s) - 6, ax.Y(0) + 16)
		if (a0 !== a) ctx.fillText(T('dashed: tuned at 5 ms only', 'kesikli: yalnız 5 ms’de ayarlı'), ax.X(52), ax.Y(YM) + 14)

		rRange(w === 0 ? '5 ms' : `5–${num(5 + 40 * w, 0)} ms`)
		rKp(num(KP[a], 0))
		rNom(`${num(at(a, 5), 2)}°`)
		rReal(`${eReal >= CAP ? '> 20' : num(eReal, 2)}° ${eReal < 2 ? '✓' : '✗'}`)
	})
}
