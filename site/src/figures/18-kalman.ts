// Lesson 18 · a constant-velocity Kalman filter on the lesson-17 signal (0.5 sin(2π 0.2 t) +
// 0.2 sin(2π 0.5 t) rad, 2 ms sampling, 4096-count encoder, 10 mrad white noise, fixed seed).
// Knobs: the process noise q_c and the R the filter is told (as a multiple of the true R).
// Optional: a 0.3 rad step at t = 5 s, drawn as a critically damped response with a peak speed
// near the 4.05 rad/s of the lesson's step test. Re-run over the 10 s whenever a control changes.
import { animate, axes, colors, frame, mono, rng, type Lang } from './_kit'

const DT = 0.002, N = 5000, SIGMA = 0.01, STEP = (2 * Math.PI) / 4096, SKIP = 250
const R_TRUE = SIGMA * SIGMA + (STEP * STEP) / 12
const W_STEP = (4.05 * Math.E) / 0.3 // peak speed of 0.3 (1 − (1 + ωt) e^(−ωt)) is 0.3 ω / e

const noise = (() => { const r = rng(18); return Array.from({ length: N }, () => r.gauss()) })()

function run(qc: number, rScale: number, step: boolean) {
	const q = new Float64Array(N), qd = new Float64Array(N)
	const xq = new Float64Array(N), xv = new Float64Array(N), sd = new Float64Array(N)
	const w1 = 2 * Math.PI * 0.2, w2 = 2 * Math.PI * 0.5
	for (let i = 0; i < N; i++) {
		const t = i * DT
		q[i] = 0.5 * Math.sin(w1 * t) + 0.2 * Math.sin(w2 * t)
		qd[i] = 0.5 * w1 * Math.cos(w1 * t) + 0.2 * w2 * Math.cos(w2 * t)
		if (step && t > 5) {
			const s = t - 5, e = Math.exp(-W_STEP * s)
			q[i] += 0.3 * (1 - (1 + W_STEP * s) * e)
			qd[i] += 0.3 * W_STEP * W_STEP * s * e
		}
	}
	const r = R_TRUE * rScale
	const Q00 = (qc * DT ** 3) / 3, Q01 = (qc * DT ** 2) / 2, Q11 = qc * DT
	let a = 0, b = 0, P00 = r, P01 = 0, P11 = 1, K0 = 0, K1 = 0
	for (let i = 0; i < N; i++) {
		const z = Math.round(q[i] / STEP) * STEP + SIGMA * noise[i]
		if (i === 0) { a = z } else {
			// predict: x = F x, P = F P Fᵀ + Q with F = [[1, dt], [0, 1]]
			a += DT * b
			const p00 = P00 + 2 * DT * P01 + DT * DT * P11 + Q00
			const p01 = P01 + DT * P11 + Q01
			const p11 = P11 + Q11
			// update with H = [1, 0]
			const S = p00 + r
			K0 = p00 / S; K1 = p01 / S
			const innov = z - a
			a += K0 * innov; b += K1 * innov
			P00 = (1 - K0) * p00; P01 = (1 - K0) * p01; P11 = p11 - K1 * p01
		}
		xq[i] = a; xv[i] = b; sd[i] = Math.sqrt(Math.max(P00, 0))
	}
	let sp = 0, sv = 0, inside = 0, maxStep = 0
	for (let i = SKIP; i < N; i++) {
		const e = xq[i] - q[i]
		sp += e * e; sv += (xv[i] - qd[i]) ** 2
		if (Math.abs(e) <= 2 * sd[i]) inside++
		if (i * DT > 5 && i * DT <= 5.5) maxStep = Math.max(maxStep, Math.abs(e))
	}
	const n = N - SKIP
	return { q, qd, xq, xv, sd, pos: Math.sqrt(sp / n), vel: Math.sqrt(sv / n), cover: (100 * inside) / n, K0, K1, maxStep }
}

export default function (root: HTMLElement, lang: Lang) {
	const T_ = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => (lang === 'tr' ? x.toFixed(d).replace('.', ',') : x.toFixed(d))
	const sci = (x: number) => (x >= 1 ? num(x, x < 10 ? 1 : 0) : num(x, Math.min(3, Math.ceil(-Math.log10(x)))))
	const f = frame(root, lang, {
		title: { en: 'Trust the model or the encoder?', tr: 'Modele mi güvenmeli, encoder’a mı?' },
		caption: {
			en: 'Top: position error of the raw reading (grey) and of the Kalman estimate (orange); the shaded band is the filter’s own ±2σ claim. Bottom: true velocity (dashed) and the estimate. Small q_c trusts the constant-velocity model: smooth, but late and overconfident. Large q_c follows the noise. Then tell the filter a wrong R, or add the step it cannot predict.',
			tr: 'Üstte: ham okumanın (gri) ve Kalman tahmininin (turuncu) konum hatası; gölgeli bant filtrenin kendi ±2σ iddiası. Altta: gerçek hız (kesikli) ve tahmin. Küçük q_c constant-velocity modeline güvenir: yumuşak ama geç ve aşırı emin. Büyük q_c noise’u izler. Sonra filtreye yanlış bir R söyle ya da öngöremeyeceği basamağı ekle.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 360)
	const logQ = f.slider({ en: 'process noise q_c', tr: 'process noise q_c' }, { min: -3, max: 4, step: 0.1, value: 0, fmt: (x) => sci(10 ** x) })
	const logR = f.slider({ en: 'R told to the filter ÷ true R', tr: 'filtreye söylenen R ÷ gerçek R' }, { min: -2, max: 2, step: 0.1, value: 0, fmt: (x) => sci(10 ** x) })
	const step = f.checkbox({ en: '0.3 rad step at t = 5 s', tr: 't = 5 s’de 0,3 rad basamak' })
	const rPos = f.readout({ en: 'position RMS', tr: 'konum RMS' })
	const rVel = f.readout({ en: 'velocity RMS', tr: 'hız RMS' })
	const rCov = f.readout({ en: '2σ coverage', tr: '2σ coverage' })
	const rK = f.readout({ en: 'gains K_q / K_q̇', tr: 'gain’ler K_q / K_q̇' })
	const rStep = f.readout({ en: 'max error 0.5 s after step', tr: 'basamaktan sonraki 0,5 s’de en büyük hata' })

	let res = run(10 ** logQ.get(), 10 ** logR.get(), step.get())
	const rerun = () => { res = run(10 ** logQ.get(), 10 ** logR.get(), step.get()) }
	logQ.on(rerun); logR.on(rerun); step.on(rerun)

	let clock = 0
	animate(root, (dt) => {
		const c = colors()
		clock = (clock + dt) % (N * DT - 1 - SKIP * DT)
		const i1 = SKIP + Math.floor(clock / DT) + 500, i0 = i1 - 500
		ctx.clearRect(0, 0, W, H)
		ctx.font = mono

		// top: position error with the claimed ±2σ band, in mrad
		const span = 40
		const a = axes(ctx, { x0: 60, y0: 14, x1: W - 14, y1: 160 }, [i0 * DT, i1 * DT], [-span, span])
		a.draw(c, '', T_('position error [mrad]', 'konum hatası [mrad]'))
		ctx.fillStyle = c.soft; ctx.beginPath()
		for (let i = i0; i < i1; i++) ctx.lineTo(a.X(i * DT), a.Y(Math.min(span, 2000 * res.sd[i])))
		for (let i = i1 - 1; i >= i0; i--) ctx.lineTo(a.X(i * DT), a.Y(Math.max(-span, -2000 * res.sd[i])))
		ctx.closePath(); ctx.fill()
		a.line([[i0 * DT, 0], [i1 * DT, 0]], c.line, 1)
		const raw: [number, number][] = [], est: [number, number][] = []
		for (let i = i0; i < i1; i++) {
			const z = Math.round(res.q[i] / STEP) * STEP + SIGMA * noise[i]
			raw.push([i * DT, Math.max(-span, Math.min(span, 1000 * (z - res.q[i])))])
			est.push([i * DT, Math.max(-span, Math.min(span, 1000 * (res.xq[i] - res.q[i])))])
		}
		a.line(raw, c.line, 1)
		a.line(est, c.accent, 2.5)
		ctx.fillStyle = c.muted
		ctx.fillText(String(span), 56 - ctx.measureText(String(span)).width, 26)
		ctx.fillText('−' + span, 56 - ctx.measureText('−' + span).width, 160)
		const tl = `t = ${num(i0 * DT, 1)}–${num(i1 * DT, 1)} s`
		ctx.fillText(tl, W - 14 - ctx.measureText(tl).width, 30)

		// bottom: velocity, true vs estimated
		const vs = step.get() ? 4.8 : 1.6
		const b = axes(ctx, { x0: 60, y0: 186, x1: W - 14, y1: H - 30 }, [i0 * DT, i1 * DT], [-vs, vs])
		b.draw(c, T_('time [s] →', 'zaman [s] →'), T_('velocity [rad/s]', 'hız [rad/s]'))
		const vt: [number, number][] = [], ve: [number, number][] = []
		for (let i = i0; i < i1; i++) {
			vt.push([i * DT, res.qd[i]])
			ve.push([i * DT, Math.max(-vs, Math.min(vs, res.xv[i]))])
		}
		b.line(vt, c.ink, 1.5, [5, 4])
		b.line(ve, c.accent, 2.5)
		ctx.fillStyle = c.muted
		const vl = num(vs, 1)
		ctx.fillText(vl, 56 - ctx.measureText(vl).width, 198)
		ctx.fillText('−' + vl, 56 - ctx.measureText('−' + vl).width, H - 30)

		rPos(num(res.pos * 1000, 2) + ' mrad')
		rVel(num(res.vel, 3) + ' rad/s')
		rCov(lang === 'tr' ? '%' + num(res.cover, 1) : num(res.cover, 1) + '%')
		rK(`${num(res.K0, 4)} / ${num(res.K1, 3)}`)
		rStep(step.get() ? num(res.maxStep * 1000, 1) + ' mrad' : '—')
	})
}
