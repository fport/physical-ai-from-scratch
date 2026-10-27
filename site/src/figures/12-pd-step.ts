// Lesson 12 · a single link held out against gravity by a PD controller, with optional gravity
// feedforward and a torque cap. Point mass m at length l (the lesson's part-B link: 1 kg, 30 cm),
// semi-implicit Euler at 1 ms.
import { animate, axes, colors, frame, mono, type Lang } from './_kit'

export default function (root: HTMLElement, lang: Lang) {
	const f = frame(root, lang, {
		title: { en: 'PD on a link that gravity pulls down', tr: 'Yerçekiminin aşağı çektiği bir link’te PD' },
		caption: {
			en: 'Change Kp and Kd and watch the angle settle. With PD alone the link settles below the target, because the spring Kp·e has to hold the weight. Gravity feedforward adds that torque directly and the sag disappears.',
			tr: 'Kp ve Kd’yi değiştir, açının nasıl oturduğunu izle. Yalnız PD ile link hedefin altında kalır, çünkü ağırlığı Kp·e yayı taşımak zorunda. Gravity feedforward bu torque’u doğrudan ekler ve sarkma kaybolur.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 280)
	const kp = f.slider({ en: 'Kp [N·m/rad]', tr: 'Kp [N·m/rad]' }, { min: 5, max: 200, step: 1, value: 40 })
	const kd = f.slider({ en: 'Kd [N·m·s/rad]', tr: 'Kd [N·m·s/rad]' }, { min: 0, max: 10, step: 0.1, value: 1.5, fmt: (x) => x.toFixed(1) })
	const ff = f.checkbox({ en: 'gravity feedforward', tr: 'gravity feedforward' })
	const cap = f.checkbox({ en: 'torque cap 3 N·m', tr: 'torque cap 3 N·m' })
	const rSag = f.readout({ en: 'sag', tr: 'sarkma' })
	const rOver = f.readout({ en: 'overshoot', tr: 'overshoot' })
	const rZeta = f.readout({ en: 'ζ', tr: 'ζ' })

	const m = 1, l = 0.3, g = 9.81, I = m * l * l, mgl = m * g * l, h = 0.001
	const targets = [0.2, 1.0]
	let ti = 1, q = targets[0], dq = 0, t = 0, start = q, peak = q
	let hist: [number, number][] = []
	const restart = () => { start = q; peak = q; t = 0; hist = [] }
	f.button({ en: 'New target', tr: 'Yeni hedef' }, () => { ti = 1 - ti; restart() })
	for (const c of [kp, kd]) c.on(restart)
	for (const c of [ff, cap]) c.on(restart)

	animate(root, (dt) => {
		const target = targets[ti]
		const steps = Math.round(Math.min(dt, 0.05) / h)
		for (let i = 0; i < steps && t < 4; i++) {
			const grav = mgl * Math.cos(q)
			let tau = kp.get() * (target - q) - kd.get() * dq + (ff.get() ? grav : 0)
			if (cap.get()) tau = Math.max(-3, Math.min(3, tau))
			dq += (h * (tau - grav)) / I
			q += h * dq
			t += h
			peak = target > start ? Math.max(peak, q) : Math.min(peak, q)
			if (i % 10 === 0) hist.push([t, q])
		}
		const c = colors()
		ctx.clearRect(0, 0, W, H)
		// the link
		const px = 120, py = 175, L = 120
		ctx.strokeStyle = c.line; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4])
		ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px + L * Math.cos(target), py - L * Math.sin(target)); ctx.stroke(); ctx.setLineDash([])
		ctx.strokeStyle = c.ink; ctx.lineWidth = 10; ctx.lineCap = 'round'
		ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px + L * Math.cos(q), py - L * Math.sin(q)); ctx.stroke()
		ctx.fillStyle = c.accent; ctx.beginPath(); ctx.arc(px + L * Math.cos(q), py - L * Math.sin(q), 11, 0, 7); ctx.fill()
		ctx.fillStyle = c.bg; ctx.strokeStyle = c.ink; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(px, py, 8, 0, 7); ctx.fill(); ctx.stroke()
		ctx.fillStyle = c.muted; ctx.font = mono
		ctx.fillText('↓ g', px - 10, py + 55)
		ctx.fillText(lang === 'tr' ? 'kesikli = hedef' : 'dashed = target', 16, 22)
		// angle over time
		const a = axes(ctx, { x0: 270, y0: 20, x1: W - 16, y1: H - 30 }, [0, 4], [-0.2, 1.3])
		a.draw(c, lang === 'tr' ? 'zaman [s] →' : 'time [s] →', lang === 'tr' ? 'açı [rad]' : 'angle [rad]')
		a.line([[0, target], [4, target]], c.line, 1.5, [5, 5])
		a.line(hist, c.accent)
		const span = target - start
		rSag((((target - q) * 180) / Math.PI).toFixed(2) + '°')
		rOver(Math.abs(span) > 0.05 ? Math.max(0, ((peak - target) * Math.sign(span)) / Math.abs(span) * 100).toFixed(0) + '%' : '—')
		rZeta((kd.get() / (2 * Math.sqrt(kp.get() * I))).toFixed(2))
	})
}
