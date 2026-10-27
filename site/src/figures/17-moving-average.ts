// Lesson 17 · the moving-average trade-off on the lesson's signal: shoulder_pan follows
// 0.5 sin(2π 0.2 k t) + 0.2 sin(2π 0.5 k t) rad, sampled at 2 ms by a 4096-count encoder with
// 10 mrad white noise (fixed seed). k speeds the motion up as in "Break it". Velocity is the
// central difference of the filtered signal, like np.gradient. The whole 10 s run is recomputed
// for every window 1–129 when a control changes; errors skip the first 0.5 s.
import { animate, axes, colors, frame, mono, rng, type Lang } from './_kit'

const DT = 0.002, N = 5000, SIGMA = 0.01, STEP = (2 * Math.PI) / 4096, NMAX = 129, SKIP = 250
const SPEEDS = [0.5, 1, 2, 4, 8]

const noise = (() => { const r = rng(17); return Array.from({ length: N }, () => r.gauss()) })()

function run(k: number) {
	const q = new Float64Array(N), qd = new Float64Array(N), z = new Float64Array(N)
	const w1 = 2 * Math.PI * 0.2 * k, w2 = 2 * Math.PI * 0.5 * k
	for (let i = 0; i < N; i++) {
		const t = i * DT
		q[i] = 0.5 * Math.sin(w1 * t) + 0.2 * Math.sin(w2 * t)
		qd[i] = 0.5 * w1 * Math.cos(w1 * t) + 0.2 * w2 * Math.cos(w2 * t)
		z[i] = Math.round(q[i] / STEP) * STEP + SIGMA * noise[i]
	}
	return { q, qd, z }
}

function movingAverage(z: Float64Array, n: number) {
	const y = new Float64Array(z.length)
	let s = 0
	for (let i = 0; i < z.length; i++) {
		s += z[i]
		if (i >= n) s -= z[i - n]
		y[i] = s / Math.min(n, i + 1)
	}
	return y
}

function gradient(y: Float64Array) {
	const v = new Float64Array(y.length), L = y.length
	v[0] = (y[1] - y[0]) / DT
	v[L - 1] = (y[L - 1] - y[L - 2]) / DT
	for (let i = 1; i < L - 1; i++) v[i] = (y[i + 1] - y[i - 1]) / (2 * DT)
	return v
}

const rms = (e: Float64Array) => { let s = 0; for (let i = SKIP; i < e.length - 1; i++) s += e[i] * e[i]; return Math.sqrt(s / (e.length - 1 - SKIP)) }

export default function (root: HTMLElement, lang: Lang) {
	const T_ = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => (lang === 'tr' ? x.toFixed(d).replace('.', ',') : x.toFixed(d))
	const f = frame(root, lang, {
		title: { en: 'Less noise, more lag', tr: 'Daha az noise, daha çok gecikme' },
		caption: {
			en: 'Top: the error of the raw encoder reading (grey) and of its moving average (orange) over one second. Bottom: the filter’s RMS error for every window size; a window of n samples is (n−1) ms late at 2 ms sampling. Grow the window and the noise shrinks while the lag grows. Switch to velocity, or speed the motion up, and the best window moves.',
			tr: 'Üstte: ham encoder okumasının (gri) ve moving average’ının (turuncu) bir saniyelik hatası. Altta: her pencere boyu için filtrenin RMS hatası; 2 ms örneklemede n örneklik pencere (n−1) ms geç kalır. Pencereyi büyüt: noise küçülür, gecikme büyür. Hıza geç ya da hareketi hızlandır; en iyi pencere yer değiştirir.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 360)
	const win = f.slider({ en: 'window n [samples]', tr: 'pencere n [örnek]' }, { min: 1, max: NMAX, step: 1, value: 5 })
	const speed = f.slider({ en: 'motion speed k', tr: 'hareket hızı k' }, { min: 0, max: SPEEDS.length - 1, step: 1, value: 1, fmt: (i) => num(SPEEDS[i], SPEEDS[i] < 1 ? 1 : 0) + '×' })
	const mode = f.select({ en: 'estimate', tr: 'tahmin' }, [{ en: 'position', tr: 'konum' }, { en: 'velocity', tr: 'hız' }], 0)
	const rDelay = f.readout({ en: 'delay', tr: 'delay' })
	const rErr = f.readout({ en: 'RMS error', tr: 'RMS hata' })
	const rRaw = f.readout({ en: 'raw', tr: 'ham' })
	const rBest = f.readout({ en: 'best window', tr: 'en iyi pencere' })

	let sig = run(SPEEDS[speed.get()])
	let curve: number[] = [] // RMS error per window, index n-1
	let errRaw = new Float64Array(N), errFilt = new Float64Array(N), span = 1
	const recompute = () => {
		sig = run(SPEEDS[speed.get()])
		const vel = mode.get() === 1
		curve = []
		for (let n = 1; n <= NMAX; n++) {
			const y = movingAverage(sig.z, n)
			const e = new Float64Array(N)
			const est = vel ? gradient(y) : y, ref = vel ? sig.qd : sig.q
			for (let i = 0; i < N; i++) e[i] = est[i] - ref[i]
			curve.push(rms(e))
		}
		refilter()
	}
	const refilter = () => {
		const vel = mode.get() === 1
		const ref = vel ? sig.qd : sig.q
		const raw = vel ? gradient(sig.z) : sig.z
		const est = vel ? gradient(movingAverage(sig.z, win.get())) : movingAverage(sig.z, win.get())
		for (let i = 0; i < N; i++) { errRaw[i] = raw[i] - ref[i]; errFilt[i] = est[i] - ref[i] }
		let m = 0
		for (let i = SKIP; i < N - 1; i++) m = Math.max(m, Math.abs(errFilt[i]))
		span = Math.max(2.5 * curve[0], 1.15 * m)
	}
	recompute()
	speed.on(recompute); mode.on(recompute); win.on(refilter)

	let clock = 0
	animate(root, (dt) => {
		const c = colors()
		const vel = mode.get() === 1, n = win.get()
		const scale = vel ? 1 : 1000, unit = vel ? 'rad/s' : 'mrad'
		clock = (clock + dt) % (N * DT - 1 - SKIP * DT)
		const i1 = SKIP + Math.floor(clock / DT) + 500, i0 = i1 - 500
		ctx.clearRect(0, 0, W, H)
		ctx.font = mono

		// top: error over the last second
		const a = axes(ctx, { x0: 60, y0: 14, x1: W - 14, y1: 150 }, [i0 * DT, i1 * DT], [-span * scale, span * scale])
		a.draw(c, '', T_(`error [${unit}]`, `hata [${unit}]`))
		a.line([[i0 * DT, 0], [i1 * DT, 0]], c.line, 1)
		const rawPts: [number, number][] = [], filtPts: [number, number][] = []
		for (let i = i0; i < i1; i++) {
			rawPts.push([i * DT, Math.max(-span, Math.min(span, errRaw[i])) * scale])
			filtPts.push([i * DT, errFilt[i] * scale])
		}
		a.line(rawPts, c.line, 1)
		a.line(filtPts, c.accent, 2.5)
		ctx.fillStyle = c.muted
		const lab = num(span * scale, vel ? 2 : 0)
		ctx.fillText(lab, 56 - ctx.measureText(lab).width, 26)
		ctx.fillText('−' + lab, 56 - ctx.measureText('−' + lab).width, 150)

		// bottom: RMS error against window size, log-log
		const lo = Math.log10(Math.min(...curve)) - 0.15, hi = Math.log10(Math.max(...curve)) + 0.15
		const b = axes(ctx, { x0: 60, y0: 190, x1: W - 14, y1: H - 32 }, [0, Math.log10(NMAX)], [lo, hi])
		b.draw(c, T_('window n (delay n−1 ms) →', 'pencere n (delay n−1 ms) →'), T_(`RMS [${unit}]`, `RMS [${unit}]`))
		ctx.fillStyle = c.muted; ctx.strokeStyle = c.line; ctx.lineWidth = 1
		for (const t of [1, 3, 5, 9, 17, 33, 65, 129]) {
			const X = b.X(Math.log10(t))
			ctx.beginPath(); ctx.moveTo(X, H - 32); ctx.lineTo(X, H - 27); ctx.stroke()
			ctx.fillText(String(t), X - ctx.measureText(String(t)).width / 2, H - 14)
		}
		for (let d = Math.ceil(lo); d <= Math.floor(hi); d++) {
			const s = vel ? num(10 ** d, Math.max(0, -d)) : num(10 ** (d + 3), Math.max(0, -(d + 3)))
			ctx.fillText(s, 56 - ctx.measureText(s).width, b.Y(d) + 4)
		}
		b.line([[0, Math.log10(curve[0])], [Math.log10(NMAX), Math.log10(curve[0])]], c.muted, 1.2, [5, 5])
		b.line(curve.map((e, j) => [Math.log10(j + 1), Math.log10(e)] as [number, number]), c.accent, 2.5)
		let best = 0
		curve.forEach((e, j) => { if (e < curve[best]) best = j })
		ctx.strokeStyle = c.ink; ctx.lineWidth = 1.5
		ctx.beginPath(); ctx.arc(b.X(Math.log10(best + 1)), b.Y(Math.log10(curve[best])), 6, 0, 7); ctx.stroke()
		ctx.fillStyle = c.accent
		ctx.beginPath(); ctx.arc(b.X(Math.log10(n)), b.Y(Math.log10(curve[n - 1])), 6, 0, 7); ctx.fill()
		ctx.fillStyle = c.muted
		ctx.fillText(T_('dashed = no filter · ring = best', 'kesikli = filtresiz · halka = en iyi'), b.X(0) + 110, 202)

		rDelay(`${n - 1} ms`)
		rErr(`${num(curve[n - 1] * scale, 2)} ${unit}`)
		rRaw(`${num(curve[0] * scale, 2)} ${unit}`)
		rBest(`n = ${best + 1} (${best} ms)`)
	})
}
