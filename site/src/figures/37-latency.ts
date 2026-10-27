// Lesson 37 · a PD joint that sees its own angle late. One joint, I = 0.0901 kg·m², no gravity, PD
// with ζ = 0.7 (Kd = 2ζ√(Kp·I)), step 0 → 0.5 rad, physics at 1 kHz (semi-implicit Euler). The
// controller runs at f Hz with zero-order hold, reads the joint d ms late, and optionally applies
// each command one tick later. The delay budget is the lesson's formula d_max = PM/ω_c.
import { animate, axes, colors, frame, mono, type Lang } from './_kit'

export default function (root: HTMLElement, lang: Lang) {
	const T = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => (lang === 'tr' ? x.toFixed(d).replace('.', ',') : x.toFixed(d))
	const pct = (x: number, d: number) => (lang === 'tr' ? `%${num(x, d)}` : `${num(x, d)}%`)
	const f = frame(root, lang, {
		title: { en: 'Delay against the loop’s own speed', tr: 'Gecikme, döngünün kendi hızına karşı' },
		caption: {
			en: 'Add measurement delay, lower the control rate or add a tick of compute, and watch the step response ring and then diverge once the total passes the delay margin PM/ω_c. Raise Kp at the same damping ratio and the margin shrinks.',
			tr: 'Ölçüm gecikmesi ekle, kontrol hızını düşür ya da bir tick compute ekle; toplam gecikme PM/ω_c delay margin’ini geçince step response önce salınır, sonra ıraksar. Aynı damping ratio’da Kp’yi artır; margin küçülür.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 290)
	const rates = [1000, 500, 200, 100, 50, 30, 20, 10]
	const delay = f.slider({ en: 'measurement delay [ms]', tr: 'ölçüm gecikmesi [ms]' }, { min: 0, max: 100, step: 1, value: 0 })
	const rate = f.slider({ en: 'control rate [Hz]', tr: 'kontrol hızı [Hz]' }, { min: 0, max: rates.length - 1, step: 1, value: 0, fmt: (i) => String(rates[i]) })
	const kp = f.slider({ en: 'Kp [N·m/rad], ζ = 0.7', tr: 'Kp [N·m/rad], ζ = 0,7' }, { min: 5, max: 160, step: 1, value: 20 })
	const tick = f.checkbox({ en: '+1 tick of compute', tr: '+1 tick compute' })
	const rMargin = f.readout({ en: 'delay margin PM/ω_c', tr: 'delay margin PM/ω_c' })
	const rTotal = f.readout({ en: 'total delay', tr: 'toplam gecikme' })
	const rOver = f.readout({ en: 'overshoot', tr: 'overshoot' })

	const I = 0.0901, ZETA = 0.7, TARGET = 0.5, h = 0.001, TEND = 4, N = Math.round(TEND / h)
	const q = new Float64Array(N), v = new Float64Array(N)
	let nValid = N, diverged = false, unstable = false, over = 0
	let wc = 0, dmax = 0, dtot = 0

	const simulate = () => {
		const Kp = kp.get(), Kd = 2 * ZETA * Math.sqrt(Kp * I), fr = rates[rate.get()], nd = delay.get()
		// analytic crossover and margin of L(s) = (Kp + Kd s) / (I s²)
		wc = Math.sqrt((Kd ** 2 + Math.sqrt(Kd ** 4 + 4 * I * I * Kp * Kp)) / (2 * I * I))
		dmax = Math.atan((Kd * wc) / Kp) / wc
		dtot = nd / 1000 + 0.5 / fr + (tick.get() ? 1 / fr : 0)
		let x = 0, xd = 0, u = 0, pending = 0, last = -1
		diverged = false; unstable = false; nValid = N; over = 0
		for (let k = 0; k < N; k++) {
			q[k] = x; v[k] = xd
			const t = Math.floor((k * fr) / 1000)
			if (t !== last) {
				last = t
				const j = Math.max(0, k - nd)
				const un = Kp * (TARGET - q[j]) - Kd * v[j]
				if (tick.get()) { u = pending; pending = un } else u = un
			}
			xd += (h * u) / I
			x += h * xd
			over = Math.max(over, x - TARGET)
			if (Math.abs(x) > 5) { diverged = true; nValid = k + 1; break }
		}
		// a loop that still swings by more than half the step in the last 0.8 s is unstable
		if (!diverged) {
			let late = 0
			for (let k = Math.floor(N * 0.8); k < N; k++) late = Math.max(late, Math.abs(q[k] - TARGET))
			unstable = late > 0.5 * TARGET
		}
	}
	simulate()
	for (const c of [delay, rate, kp]) c.on(simulate)
	tick.on(simulate)

	let cursor = 0
	animate(root, (dt) => {
		cursor = (cursor + dt) % (TEND + 0.8)
		const c = colors()
		ctx.clearRect(0, 0, W, H)
		const kc = Math.max(0, Math.min(nValid - 1, Math.floor(Math.min(cursor, TEND) / h)))
		// left: the joint (angle exaggerated 1.6× for visibility), target dashed
		const px = 100, py = 150, L = 80
		const ang = (th: number) => -Math.PI / 2 + Math.max(-3, Math.min(3, th)) * 1.6
		ctx.strokeStyle = c.line; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4])
		ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px + L * Math.cos(ang(TARGET)), py + L * Math.sin(ang(TARGET))); ctx.stroke(); ctx.setLineDash([])
		ctx.strokeStyle = c.ink; ctx.lineWidth = 9; ctx.lineCap = 'round'
		const qa = ang(q[kc])
		ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px + L * Math.cos(qa), py + L * Math.sin(qa)); ctx.stroke()
		ctx.fillStyle = c.accent; ctx.beginPath(); ctx.arc(px + L * Math.cos(qa), py + L * Math.sin(qa), 9, 0, 7); ctx.fill()
		ctx.fillStyle = c.bg; ctx.strokeStyle = c.ink; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(px, py, 7, 0, 7); ctx.fill(); ctx.stroke()
		ctx.lineCap = 'butt'
		ctx.fillStyle = c.muted; ctx.font = mono
		ctx.fillText(T('dashed = target', 'kesikli = hedef'), 16, 22)
		// delay budget bar
		const bx = 16, by = 246, bw = 200, used = dtot / dmax
		ctx.strokeStyle = c.line; ctx.lineWidth = 1.5; ctx.strokeRect(bx, by, bw, 12)
		ctx.fillStyle = used > 1 ? c.accent : c.ink
		ctx.globalAlpha = used > 1 ? 1 : 0.55
		ctx.fillRect(bx, by, bw * Math.min(1, used), 12)
		ctx.globalAlpha = 1
		ctx.fillStyle = c.muted
		ctx.fillText(T(`delay budget used: ${pct(100 * used, 0)}`, `kullanılan gecikme bütçesi: ${pct(100 * used, 0)}`), bx, by - 8)
		// right: angle over time
		const a = axes(ctx, { x0: 260, y0: 24, x1: W - 16, y1: H - 34 }, [0, TEND], [-0.25, 1.4])
		a.draw(c, T('time [s] →', 'zaman [s] →'), T('angle [rad]', 'açı [rad]'))
		a.line([[0, TARGET], [TEND, TARGET]], c.line, 1.5, [5, 5])
		const pts: [number, number][] = []
		for (let k = 0; k < nValid; k += 5) pts.push([k * h, Math.max(-0.25, Math.min(1.4, q[k]))])
		ctx.save(); ctx.beginPath(); ctx.rect(260, 24, W - 276, H - 58); ctx.clip()
		a.line(pts, c.accent)
		ctx.restore()
		ctx.strokeStyle = c.muted; ctx.lineWidth = 1
		ctx.beginPath(); ctx.moveTo(a.X(kc * h), a.Y(1.4)); ctx.lineTo(a.X(kc * h), a.Y(-0.25)); ctx.stroke()
		ctx.fillStyle = c.muted
		for (const s of [0, 1, 2, 3]) ctx.fillText(String(s), a.X(s) - 4, a.Y(-0.25) + 16)
		if (diverged || unstable) { ctx.fillStyle = c.accent; ctx.fillText(diverged ? T('diverges', 'ıraksıyor') : T('unstable', 'kararsız'), a.X(0.1), a.Y(1.25)) }

		rMargin(`${num(dmax * 1000, 1)} ms (ω_c ${num(wc, 2)} rad/s)`)
		rTotal(`${num(dtot * 1000, 1)} ms`)
		rOver(diverged ? T('diverges', 'ıraksıyor') : pct((100 * over) / TARGET, 1) + (unstable ? T(' (unstable)', ' (kararsız)') : ''))
	})
}
