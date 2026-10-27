// Lesson 24 · LQR on the lesson's cart-pole (cart 1 kg, pole 0.2 kg / 0.6 m as a uniform rod, track
// ±1.5 m). The Euler step at dt = 0.01 s is linearized by finite differences at the upright, the
// discrete Riccati equation is iterated to convergence, and u = clip(−K δx, ±F) is applied at 100 Hz
// while the plant is integrated at 1 ms. Failure modes as in the lesson: the pole falls (force
// saturated) or the cart runs into the track end (LQR only penalizes x, it does not know the wall).
import { animate, axes, colors, frame, mono, type Lang } from './_kit'

type V4 = [number, number, number, number]
const I4 = [0, 1, 2, 3]

export default function (root: HTMLElement, lang: Lang) {
	const f = frame(root, lang, {
		title: { en: 'How far from upright does LQR still work?', tr: 'LQR dikten ne kadar uzakta hâlâ çalışır?' },
		caption: {
			en: 'Release the pole at an angle and let a fixed LQR gain catch it. Lower the force limit and the command saturates: the pole falls. Raise it and the pole is caught, but the cart may run into the end of the track. Q on cart position and R on force reshape K.',
			tr: 'Çubuğu bir açıdan bırak, sabit bir LQR gain’i yakalasın. Kuvvet limitini düşür: komut doyuma ulaşır ve çubuk düşer. Yükselt: çubuk yakalanır ama araba rayın sonuna çarpabilir. Araba pozisyonundaki Q ve kuvvetteki R, K’yı yeniden şekillendirir.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 320)
	const th0 = f.slider({ en: 'release angle [°]', tr: 'bırakma açısı [°]' }, { min: 0, max: 60, step: 5, value: 25 })
	const Fmax = f.slider({ en: 'force limit F [N]', tr: 'kuvvet limiti F [N]' }, { min: 5, max: 50, step: 1, value: 10 })
	const qx = f.slider({ en: 'Q on cart position', tr: 'araba pozisyonunda Q' }, { min: -1, max: 2, step: 0.1, value: 0, fmt: (x) => fmtG(lang, 10 ** x) })
	const rr = f.slider({ en: 'R on force', tr: 'kuvvette R' }, { min: -3, max: -1, step: 0.1, value: -2, fmt: (x) => fmtG(lang, 10 ** x) })
	const rStat = f.readout({ en: 'status', tr: 'durum' })
	const rU = f.readout({ en: 'force', tr: 'kuvvet' })
	const rK = f.readout({ en: 'K', tr: 'K' })

	// cart-pole dynamics, θ = 0 upright, state (x, θ, ẋ, θ̇)
	const M = 1, m = 0.2, lh = 0.3, g = 9.81, mt = M + m, TRACK = 1.5
	const deriv = (s: V4, u: number): V4 => {
		const sn = Math.sin(s[1]), cs = Math.cos(s[1])
		const temp = (u + m * lh * s[3] * s[3] * sn) / mt
		const thdd = (g * sn - cs * temp) / (lh * (4 / 3 - (m * cs * cs) / mt))
		return [s[2], s[3], temp - (m * lh * thdd * cs) / mt, thdd]
	}
	const DT = 0.01
	const euler = (s: V4, u: number): V4 => { const d = deriv(s, u); return I4.map((i) => s[i] + DT * d[i]) as V4 }

	const lqr = (Q: V4, R: number): V4 => {
		const e = 1e-6, z: V4 = [0, 0, 0, 0]
		const A = I4.map(() => [0, 0, 0, 0])
		for (const j of I4) {
			const sp = z.slice() as V4, sm = z.slice() as V4
			sp[j] += e; sm[j] -= e
			const a = euler(sp, 0), b = euler(sm, 0)
			for (const i of I4) A[i][j] = (a[i] - b[i]) / (2 * e)
		}
		const bp = euler(z, e), bm = euler(z, -e)
		const B = I4.map((i) => (bp[i] - bm[i]) / (2 * e))
		let P = I4.map((i) => I4.map((j) => (i === j ? Q[i] : 0)))
		let K = [0, 0, 0, 0]
		for (let it = 0; it < 20000; it++) {
			const PB = I4.map((i) => I4.reduce((s, j) => s + P[i][j] * B[j], 0))
			const BPB = I4.reduce((s, i) => s + B[i] * PB[i], 0)
			const BPA = I4.map((j) => I4.reduce((s, i) => s + PB[i] * A[i][j], 0))
			K = BPA.map((v) => v / (R + BPB))
			const PA = I4.map((i) => I4.map((j) => I4.reduce((s, k) => s + P[i][k] * A[k][j], 0)))
			const N = I4.map((i) => I4.map((j) => (i === j ? Q[i] : 0) + I4.reduce((s, k) => s + A[k][i] * PA[k][j], 0) - BPA[i] * K[j]))
			let d = 0
			for (const i of I4) for (const j of I4) d = Math.max(d, Math.abs(N[i][j] - P[i][j]) / (1 + Math.abs(P[i][j])))
			P = I4.map((i) => I4.map((j) => (N[i][j] + N[j][i]) / 2)) // keep P symmetric, or round-off grows
			if (d < 1e-10) break
		}
		return K as V4
	}

	let K = lqr([1, 10, 0.1, 0.1], 0.01)
	let s: V4 = [0, 0, 0, 0], t = 0, u = 0, status: 'run' | 'fell' | 'track' = 'run'
	let hist: [number, number][] = [], ctrlClock = 0
	const release = () => {
		s = [0, (th0.get() * Math.PI) / 180, 0, 0]; t = 0; u = 0; status = 'run'; hist = []; ctrlClock = 0
	}
	const regain = () => { K = lqr([10 ** qx.get(), 10, 0.1, 0.1], 10 ** rr.get()); release() }
	release()
	f.button({ en: 'Release again', tr: 'Yeniden bırak' }, release)
	f.button({ en: 'Push the pole', tr: 'Çubuğu it' }, () => { if (status === 'run') s[3] += 1.0 })
	th0.on(release); Fmax.on(release)
	qx.on(regain); rr.on(regain)

	animate(root, (dt) => {
		const steps = Math.round(Math.min(dt, 0.05) / 0.001)
		for (let i = 0; i < steps && status === 'run'; i++) {
			if (ctrlClock <= 0) {
				u = -I4.reduce((acc, j) => acc + K[j] * s[j], 0)
				u = Math.max(-Fmax.get(), Math.min(Fmax.get(), u))
				ctrlClock = DT
			}
			const d = deriv(s, u)
			for (const j of I4) s[j] += 0.001 * d[j]
			ctrlClock -= 0.001; t += 0.001
			if (Math.abs(s[0]) > TRACK) { s[0] = Math.sign(s[0]) * TRACK; status = 'track' }
			else if (Math.abs(s[1]) > Math.PI / 2) status = 'fell'
			if (i % 20 === 0) { hist.push([t, (s[1] * 180) / Math.PI]); if (hist.length > 400) hist.shift() }
		}
		const c = colors()
		ctx.clearRect(0, 0, W, H)
		const S = 180, cx = W / 2, ty = 190
		const PX = (x: number) => cx + x * S
		// track and its ends
		ctx.strokeStyle = c.line; ctx.lineWidth = 3
		ctx.beginPath(); ctx.moveTo(PX(-TRACK) - 14, ty); ctx.lineTo(PX(TRACK) + 14, ty); ctx.stroke()
		ctx.strokeStyle = c.ink; ctx.lineWidth = 4
		for (const e of [-TRACK, TRACK]) { const x = PX(e) + Math.sign(e) * 12; ctx.beginPath(); ctx.moveTo(x, ty - 30); ctx.lineTo(x, ty + 6); ctx.stroke() }
		// cart, pole, force arrow
		const cxp = PX(s[0])
		ctx.fillStyle = c.panel; ctx.strokeStyle = c.ink; ctx.lineWidth = 2
		ctx.fillRect(cxp - 30, ty - 26, 60, 22); ctx.strokeRect(cxp - 30, ty - 26, 60, 22)
		const L = 0.6 * S, px = cxp + L * Math.sin(s[1]), py = ty - 26 - L * Math.cos(s[1])
		ctx.strokeStyle = c.ink; ctx.lineWidth = 7; ctx.lineCap = 'round'
		ctx.beginPath(); ctx.moveTo(cxp, ty - 26); ctx.lineTo(px, py); ctx.stroke()
		ctx.fillStyle = c.accent; ctx.beginPath(); ctx.arc(px, py, 7, 0, 7); ctx.fill()
		const sat = Math.abs(u) >= Fmax.get() - 1e-9
		if (Math.abs(u) > 0.05 && status === 'run') {
			const len = (u / 50) * 90, ay = ty - 15, x0 = cxp + Math.sign(u) * 32
			ctx.strokeStyle = sat ? c.accent : c.muted; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 3
			ctx.beginPath(); ctx.moveTo(x0, ay); ctx.lineTo(x0 + len, ay); ctx.stroke()
			ctx.beginPath(); ctx.moveTo(x0 + len + Math.sign(u) * 7, ay); ctx.lineTo(x0 + len, ay - 5); ctx.lineTo(x0 + len, ay + 5); ctx.fill()
		}
		ctx.fillStyle = c.muted; ctx.font = mono
		ctx.fillText(lang === 'tr' ? '±1,5 m ray' : '±1.5 m track', PX(-TRACK), ty + 22)
		// angle over the last seconds
		const tEnd = Math.max(5, t)
		const a = axes(ctx, { x0: 50, y0: 228, x1: W - 16, y1: H - 22 }, [tEnd - 5, tEnd], [-90, 90])
		a.draw(c, lang === 'tr' ? 'zaman [s] →' : 'time [s] →', lang === 'tr' ? 'açı [°]' : 'angle [°]')
		a.line([[tEnd - 5, 0], [tEnd, 0]], c.line, 1, [4, 4])
		a.line(hist.filter(([ht]) => ht >= tEnd - 5), c.accent, 2)
		const deg = (s[1] * 180) / Math.PI
		rStat(status === 'fell' ? (lang === 'tr' ? 'düştü' : 'fell')
			: status === 'track' ? (lang === 'tr' ? 'ray sonu' : 'hit the track end')
			: t < 5 ? (lang === 'tr' ? 'yakalıyor…' : 'catching…')
			: Math.abs(deg) < 2 ? (lang === 'tr' ? 'dik (|θ| < 2°)' : 'upright (|θ| < 2°)') : (lang === 'tr' ? 'salınıyor' : 'swinging'))
		rU(fmtD(lang, u, 1) + ' N' + (sat && status === 'run' ? (lang === 'tr' ? ' (doyumda)' : ' (saturated)') : ''))
		rK('[' + K.map((k) => fmtD(lang, k, 1)).join('; ') + ']')
	})
}

const fmtG = (lang: Lang, x: number) => fmtD(lang, x, x >= 10 ? 0 : x >= 1 ? 1 : x >= 0.01 ? 2 : 3)
const fmtD = (lang: Lang, x: number, d: number) => { const s = x.toFixed(d); return lang === 'tr' ? s.replace('.', ',') : s }
