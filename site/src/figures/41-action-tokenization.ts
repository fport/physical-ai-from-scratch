// Lesson 41 · action tokens as a quantizer. One joint-target trajectory shaped like a scripted pick
// (min-jerk moves with holds, 229 samples at 50 Hz = 4.58 s, about one radian of range) is binned
// OpenVLA-style: clip to the 1st..99th percentile of the training actions, B uniform bins, decode to
// the bin centre. The error is at most Δ/2; for values spread evenly inside a bin its RMS is Δ/√12.
import { animate, axes, colors, frame, mono, type Lang } from './_kit'

export default function (root: HTMLElement, lang: Lang) {
	const T = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => (lang === 'tr' ? x.toFixed(d).replace('.', ',') : x.toFixed(d))
	const f = frame(root, lang, {
		title: { en: 'Rounding a joint target to one of B tokens', tr: 'Bir joint hedefini B token’dan birine yuvarlamak' },
		caption: {
			en: 'Thin line: the continuous joint target. Thick line: what comes back after tokenizing and decoding. Each bit removed doubles the step size and the error. Tick the box to run a motion that goes past the range frozen from the training data: there the error is no longer bounded by Δ/2.',
			tr: 'İnce çizgi: sürekli joint hedefi. Kalın çizgi: tokenize edip decode ettikten sonra geri gelen. Çıkarılan her bit adımı ve hatayı ikiye katlar. Kutuyu işaretle ve training verisinden dondurulmuş aralığın dışına çıkan bir hareket çalıştır: orada hata artık Δ/2 ile sınırlı değil.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 300)
	const bits = f.slider({ en: 'bits per dimension', tr: 'boyut başına bit' }, { min: 1, max: 10, step: 1, value: 4, fmt: (b) => `${b} (${2 ** b} ${T('bins', 'bin')})` })
	const beyond = f.checkbox({ en: 'test motion 30% larger than the training motion', tr: 'test hareketi training hareketinden %30 büyük' })
	const rDelta = f.readout({ en: 'bin width Δ', tr: 'bin genişliği Δ' })
	const rMax = f.readout({ en: 'max error (Δ/2)', tr: 'max hata (Δ/2)' })
	const rRms = f.readout({ en: 'RMS error (Δ/√12)', tr: 'RMS hata (Δ/√12)' })

	// the training trajectory: holds and min-jerk moves, [duration s, end value rad]
	const segs: [number, number][] = [[0.3, 0], [0.8, 0.55], [0.4, 0.55], [0.5, 0.85], [0.6, 0.85], [0.7, 0.35], [0.3, 0.35], [0.68, -0.15], [0.3, -0.15]]
	const n = 229, dt = 0.02
	const train = new Float64Array(n)
	{
		let t0 = 0, v0 = 0, s = 0
		for (let k = 0; k < n; k++) {
			const t = k * dt
			while (s < segs.length - 1 && t > t0 + segs[s][0]) { t0 += segs[s][0]; v0 = segs[s][1]; s++ }
			const x = Math.min(1, (t - t0) / segs[s][0])
			train[k] = v0 + (segs[s][1] - v0) * (10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5)
		}
	}
	const sorted = Array.from(train).sort((p, q) => p - q)
	const pctl = (p: number) => sorted[Math.min(n - 1, Math.max(0, Math.round((p / 100) * (n - 1))))]
	const lo = pctl(1), hi = pctl(99)
	const mid = (lo + hi) / 2
	const DEG = 180 / Math.PI

	animate(root, () => {
		const c = colors()
		const B = 2 ** bits.get(), delta = (hi - lo) / B
		const test = beyond.get() ? Array.from(train, (v) => mid + (v - mid) * 1.3) : Array.from(train)
		const dec = test.map((v) => {
			const x = (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)
			const k = Math.min(B - 1, Math.floor(x * B))
			return lo + ((k + 0.5) / B) * (hi - lo)
		})
		const err = test.map((v, i) => (dec[i] - v) * DEG)
		const emax = Math.max(...err.map(Math.abs))
		const erms = Math.sqrt(err.reduce((s, e) => s + e * e, 0) / n)

		ctx.clearRect(0, 0, W, H)
		const yr: [number, number] = [-25, 70]
		const a = axes(ctx, { x0: 56, y0: 20, x1: W - 16, y1: 182 }, [0, 4.58], yr)
		// bin edges and the training range
		ctx.strokeStyle = c.line; ctx.lineWidth = 1
		if (B <= 64) {
			ctx.beginPath()
			for (let k = 0; k <= B; k++) { const y = a.Y((lo + k * delta) * DEG); ctx.moveTo(a.X(0), y); ctx.lineTo(a.X(4.58), y) }
			ctx.globalAlpha = 0.5; ctx.stroke(); ctx.globalAlpha = 1
		}
		a.line([[0, lo * DEG], [4.58, lo * DEG]], c.muted, 1.5, [5, 4])
		a.line([[0, hi * DEG], [4.58, hi * DEG]], c.muted, 1.5, [5, 4])
		a.line(test.map((v, i) => [i * dt, v * DEG] as [number, number]), c.ink, 1.5)
		// decoded actions are held for one control step: draw them as a staircase
		const stair: [number, number][] = []
		dec.forEach((v, i) => { stair.push([i * dt, v * DEG]); stair.push([(i + 1) * dt, v * DEG]) })
		a.line(stair, c.accent, 2.5)
		a.draw(c, '', T('joint target [°]', 'joint hedefi [°]'))
		ctx.fillStyle = c.muted; ctx.font = mono
		ctx.fillText(T('dashed: 1st / 99th percentile', 'kesikli: 1. / 99. persentil'), a.X(2.35), a.Y(63))
		// error panel
		const em = Math.max(0.05, Math.min(30, Math.max(emax, delta * DEG / 2) * 1.15))
		const b = axes(ctx, { x0: 56, y0: 200, x1: W - 16, y1: H - 28 }, [0, 4.58], [-em, em])
		ctx.fillStyle = c.soft
		ctx.fillRect(b.X(0), b.Y((delta * DEG) / 2), b.X(4.58) - b.X(0), b.Y((-delta * DEG) / 2) - b.Y((delta * DEG) / 2))
		b.line([[0, 0], [4.58, 0]], c.line, 1)
		b.line(err.map((e, i) => [i * dt, e] as [number, number]), c.accent, 1.8)
		b.draw(c, T('time [s] →', 'zaman [s] →'), T('error [°], band ±Δ/2', 'hata [°], bant ±Δ/2'))
		ctx.fillStyle = c.muted
		for (const s of [0, 1, 2, 3]) ctx.fillText(String(s), b.X(s) - 4, b.Y(-em) + 16)

		rDelta(`${num(delta * DEG, delta * DEG < 1 ? 3 : 2)}°`)
		rMax(`${num(emax, 3)}° (${num((delta * DEG) / 2, 3)}°)`)
		rRms(`${num(erms, 3)}° (${num((delta * DEG) / Math.sqrt(12), 3)}°)`)
	})
}
