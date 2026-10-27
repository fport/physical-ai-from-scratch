// Lesson 43 · one-step model error compounding over an open-loop rollout. The true system is the
// lesson's pendulum (1 kg at 0.5 m, damping 0.05 N·m·s/rad, |u| ≤ 2 N·m, control step 0.05 s =
// 5 × 10 ms semi-implicit Euler, θ = 0 hanging). The "learned" model is a stand-in, not an MLP:
// the true step plus a small, smooth, state-dependent error whose size is set by a slider (the
// lesson's MLP had a one-step angle error of 0.0004 rad). Both get the same random torques (a new
// one every 5 steps, as in the lesson's planner). Optionally the model is reset to the measured
// true state every k steps, which is what replanning from measurements does.
import { animate, axes, colors, frame, mono, rng, type Lang } from './_kit'

export default function (root: HTMLElement, lang: Lang) {
	const T = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => (lang === 'tr' ? x.toFixed(d).replace('.', ',') : x.toFixed(d))
	const f = frame(root, lang, {
		title: { en: 'A small one-step error, rolled out', tr: 'Küçük bir tek adım hatası, rollout boyunca' },
		caption: {
			en: 'The model is fed its own predictions for 5 s. Start near hanging and the error grows slowly; start near upright, where nearby states fall apart, and it explodes. Re-measuring the true state every k steps resets the drift. Strong random torques swing the pendulum through both regions.',
			tr: 'Model 5 s boyunca kendi tahminleriyle beslenir. Asılı duruma yakın başlarsan hata yavaş büyür; komşu durumların birbirinden ayrıldığı dik konuma yakın başlarsan patlar. Her k adımda gerçek durumu yeniden ölçmek kaymayı sıfırlar. Güçlü rastgele torque’lar sarkacı iki bölgeden de geçirir.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 320)
	const ks = [1, 2, 5, 10, 20, 50, 100]
	const eps = f.slider({ en: 'model error per step [rad]', tr: 'adım başına model hatası [rad]' }, { min: 0, max: 0.002, step: 0.0001, value: 0.0004, fmt: (x) => num(x, 4) })
	const start = f.slider({ en: 'start angle [°] (0 = hanging)', tr: 'başlangıç açısı [°] (0 = asılı)' }, { min: 0, max: 180, step: 1, value: 10 })
	const amp = f.slider({ en: 'random torque [N·m]', tr: 'rastgele torque [N·m]' }, { min: 0, max: 2, step: 0.1, value: 0.3, fmt: (x) => num(x, 1) })
	const kSel = f.slider({ en: 're-measure every k steps', tr: 'her k adımda yeniden ölç' }, { min: 0, max: ks.length - 1, step: 1, value: ks.length - 1, fmt: (i) => (ks[i] >= 100 ? T('never', 'hiç') : String(ks[i])) })
	let seed = 3
	const rOne = f.readout({ en: 'one-step error', tr: 'tek adım hatası' })
	const r20 = f.readout({ en: 'error at 1 s', tr: '1 s’de hata' })
	const r100 = f.readout({ en: 'error at 5 s', tr: '5 s’de hata' })

	const M = 1, L = 0.5, BD = 0.05, G = 9.81, IN = M * L * L, UMAX = 2, NS = 100, DT = 0.05
	const step = (th: number, w: number, u: number): [number, number] => {
		for (let i = 0; i < 5; i++) { w += (0.01 * (u - M * G * L * Math.sin(th) - BD * w)) / IN; th += 0.01 * w }
		return [th, w]
	}
	// model = true step + e·φ(s): a smooth error that depends on the state, like a fitted network's
	const phi = (th: number, w: number) => Math.sin(2.3 * th + 0.7) * Math.cos(1.7 * w - 0.4)
	const model = (th: number, w: number, u: number, e: number): [number, number] => {
		const [a, b] = step(th, w, u), p = phi(th, w)
		return [a + e * p, b + (4 * e * p)]
	}
	const thT = new Float64Array(NS + 1), thM = new Float64Array(NS + 1), err = new Float64Array(NS + 1)
	let oneStep = 0
	const run = () => {
		const r = rng(seed)
		const U: number[] = []
		for (let i = 0; i < NS; i++) { if (i % 5 === 0) U.push((2 * r.u() - 1) * Math.min(UMAX, amp.get())); else U.push(U[i - 1]) }
		const e = eps.get(), k = ks[kSel.get()]
		let a: [number, number] = [(start.get() * Math.PI) / 180, 0], b: [number, number] = [a[0], a[1]]
		thT[0] = a[0]; thM[0] = b[0]; err[0] = 0
		let s1 = 0
		for (let i = 0; i < NS; i++) {
			if (k < 100 && i > 0 && i % k === 0) b = [a[0], a[1]] // measure the true state, restart the model there
			s1 += Math.abs(model(a[0], a[1], U[i], e)[0] - step(a[0], a[1], U[i])[0])
			a = step(a[0], a[1], U[i])
			b = model(b[0], b[1], U[i], e)
			thT[i + 1] = a[0]; thM[i + 1] = b[0]; err[i + 1] = Math.abs(b[0] - a[0])
		}
		oneStep = s1 / NS
	}
	run()
	for (const c of [eps, start, amp, kSel]) c.on(run)
	f.button({ en: 'New random torques', tr: 'Yeni rastgele torque’lar' }, () => { seed += 1; run() })

	let cursor = 0
	const fmtE = (e: number) => e === 0 ? '0 rad' : (e < 0.001 ? e.toExponential(1).replace('.', lang === 'tr' ? ',' : '.') : num(e, e < 0.1 ? 4 : 2)) + ' rad'
	animate(root, (dt) => {
		cursor = (cursor + dt) % 6
		const c = colors()
		const ic = Math.max(0, Math.min(NS, Math.floor(Math.min(cursor, 5) / DT)))
		ctx.clearRect(0, 0, W, H)
		// left: the pendulum, true (solid) and model (outline), θ = 0 hanging
		const px = 90, py = 120, R = 70
		const bob = (th: number): [number, number] => [px + R * Math.sin(th), py + R * Math.cos(th)]
		const [mx, my] = bob(thM[ic]), [tx, ty] = bob(thT[ic])
		ctx.strokeStyle = c.accent; ctx.lineWidth = 3; ctx.setLineDash([4, 3])
		ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(mx, my); ctx.stroke(); ctx.setLineDash([])
		ctx.beginPath(); ctx.arc(mx, my, 10, 0, 7); ctx.stroke()
		ctx.strokeStyle = c.ink; ctx.lineWidth = 5
		ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(tx, ty); ctx.stroke()
		ctx.fillStyle = c.ink; ctx.beginPath(); ctx.arc(tx, ty, 9, 0, 7); ctx.fill()
		ctx.fillStyle = c.bg; ctx.strokeStyle = c.ink; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(px, py, 5, 0, 7); ctx.fill(); ctx.stroke()
		ctx.fillStyle = c.muted; ctx.font = mono
		ctx.fillText(T('solid: true', 'dolu: gerçek'), 16, 232)
		ctx.fillText(T('dashed: model', 'kesikli: model'), 16, 250)
		ctx.fillText(`t = ${num(ic * DT, 2)} s`, 16, 22)
		// right top: angle
		let lo = Infinity, hi = -Infinity
		for (let i = 0; i <= NS; i++) { lo = Math.min(lo, thT[i], thM[i]); hi = Math.max(hi, thT[i], thM[i]) }
		const pad = 0.1 * (hi - lo || 1)
		const a = axes(ctx, { x0: 240, y0: 20, x1: W - 16, y1: 150 }, [0, 5], [lo - pad, hi + pad])
		a.line(Array.from(thT, (v, i) => [i * DT, v] as [number, number]), c.ink, 2)
		a.line(Array.from(thM, (v, i) => [i * DT, v] as [number, number]), c.accent, 2, [5, 3])
		a.draw(c, '', T('angle [rad]', 'açı [rad]'))
		// right bottom: |error| on a log axis
		const LMIN = -6, LMAX = 1
		const lg = (e: number) => Math.max(LMIN, Math.min(LMAX, Math.log10(e + 1e-12)))
		const b = axes(ctx, { x0: 240, y0: 172, x1: W - 16, y1: H - 30 }, [0, 5], [LMIN, LMAX])
		ctx.strokeStyle = c.line; ctx.lineWidth = 1; ctx.beginPath()
		for (let e = LMIN; e <= LMAX; e++) { ctx.moveTo(b.X(0), b.Y(e)); ctx.lineTo(b.X(5), b.Y(e)) }
		ctx.globalAlpha = 0.4; ctx.stroke(); ctx.globalAlpha = 1
		const k = ks[kSel.get()]
		if (k < 100) { ctx.strokeStyle = c.line; ctx.setLineDash([2, 3]); ctx.beginPath(); for (let i = k; i < NS; i += k) { ctx.moveTo(b.X(i * DT), b.Y(LMIN)); ctx.lineTo(b.X(i * DT), b.Y(LMAX)) } ctx.stroke(); ctx.setLineDash([]) }
		b.line(Array.from(err, (e, i) => [i * DT, lg(e)] as [number, number]).slice(1), c.accent, 2)
		b.draw(c, T('time [s] →', 'zaman [s] →'), T('|angle error| [rad]', '|açı hatası| [rad]'))
		ctx.fillStyle = c.muted
		for (const e of [-6, -4, -2, 0]) ctx.fillText(`1e${e}`, 196, b.Y(e) + 4)
		for (const s of [0, 1, 2, 3, 4]) ctx.fillText(String(s), b.X(s) - 4, b.Y(LMIN) + 16)
		// playhead
		ctx.strokeStyle = c.muted; ctx.lineWidth = 1; ctx.beginPath()
		ctx.moveTo(a.X(ic * DT), 20); ctx.lineTo(a.X(ic * DT), H - 30); ctx.stroke()

		rOne(fmtE(oneStep))
		r20(fmtE(err[20]))
		r100(fmtE(err[NS]))
	})
}
