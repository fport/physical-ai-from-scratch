// Lesson 33 · the discount γ as a horizon in seconds. A reward l policy steps ahead is weighted by
// γ^l; one step lasts Δt = 1/f, so the effective horizon is τ = Δt/(1 − γ). Pure formula, no
// simulation. Defaults are the lesson's settings: 50 Hz policy, γ = 0.95 (τ = 0.4 s).
import { animate, axes, colors, frame, mono, type Lang } from './_kit'

export default function (root: HTMLElement, lang: Lang) {
	const T = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => (lang === 'tr' ? x.toFixed(d).replace('.', ',') : x.toFixed(d))
	const pct = (x: number, d: number) => (lang === 'tr' ? `%${num(x, d)}` : `${num(x, d)}%`)
	const f = frame(root, lang, {
		title: { en: 'How far ahead does a reward still count?', tr: 'Bir reward ne kadar ileriden hâlâ sayılır?' },
		caption: {
			en: 'The bars are the weights γ^l of rewards l policy steps ahead, drawn against time in seconds. Change the control rate and the same γ covers a different stretch of time; tick the box to keep τ fixed instead (the rule lesson 34 uses). Shaded: the 1.6 s reach episode.',
			tr: 'Çubuklar, l policy adımı sonraki reward’ların ağırlıkları γ^l; yatay eksen saniye. Kontrol hızını değiştirince aynı γ başka bir zaman aralığını kapsar; kutuyu işaretlersen τ sabit kalır (ders 34’ün kuralı). Gölgeli bölge: 1,6 s’lik uzanma episode’u.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 270)
	const rates = [12.5, 25, 50, 100, 200]
	const gamma = f.slider({ en: 'γ', tr: 'γ' }, { min: 0.8, max: 0.9995, step: 0.0005, value: 0.95, fmt: (x) => num(x, 4) })
	const rate = f.slider({ en: 'policy rate [Hz]', tr: 'policy hızı [Hz]' }, { min: 0, max: rates.length - 1, step: 1, value: 2, fmt: (i) => num(rates[i], rates[i] % 1 ? 1 : 0) })
	const keep = f.checkbox({ en: 'keep τ in seconds: γ = 1 − Δt/τ', tr: 'τ’yu saniye olarak koru: γ = 1 − Δt/τ' })
	const rTau = f.readout({ en: 'horizon τ = Δt/(1−γ)', tr: 'ufuk τ = Δt/(1−γ)' })
	const rSteps = f.readout({ en: 'in steps 1/(1−γ)', tr: 'adım olarak 1/(1−γ)' })
	const rOne = f.readout({ en: 'weight of a reward 1 s ahead', tr: '1 s sonraki reward’un ağırlığı' })
	const rHalf = f.readout({ en: 'half-life', tr: 'yarı ömür' })

	// When "keep τ" is on, a rate change retunes γ so that Δt/(1−γ) stays where it was.
	let lastDt = 1 / rates[rate.get()]
	rate.on((i) => {
		const dt = 1 / rates[i]
		if (keep.get()) {
			const tau = lastDt / (1 - gamma.get())
			gamma.set(Math.min(0.9995, Math.max(0.8, 1 - dt / tau)))
		}
		lastDt = dt
	})

	const XMAX = 5
	animate(root, () => {
		const c = colors()
		const g = gamma.get(), dt = 1 / rates[rate.get()]
		const tau = dt / (1 - g)
		ctx.clearRect(0, 0, W, H)
		const a = axes(ctx, { x0: 56, y0: 30, x1: W - 16, y1: H - 34 }, [0, XMAX], [0, 1.05])
		// the 1.6 s episode of the reach task
		ctx.fillStyle = c.soft
		ctx.fillRect(a.X(0), a.Y(1.05), a.X(1.6) - a.X(0), a.Y(0) - a.Y(1.05))
		ctx.fillStyle = c.muted; ctx.font = mono
		const ep = T('episode 1.6 s', 'episode 1,6 s')
		ctx.fillText(ep, a.X(1.6) - ctx.measureText(ep).width, 18)
		// weights: one bar per policy step while bars stay visible, a filled curve otherwise
		const n = Math.floor(XMAX / dt)
		ctx.strokeStyle = c.accent; ctx.fillStyle = c.accent
		if (n <= 130) {
			ctx.lineWidth = Math.max(1.5, Math.min(5, (a.X(dt) - a.X(0)) * 0.5))
			ctx.beginPath()
			for (let l = 0; l <= n; l++) { const x = a.X(l * dt); ctx.moveTo(x, a.Y(0)); ctx.lineTo(x, a.Y(Math.pow(g, l))) }
			ctx.stroke()
		} else {
			ctx.globalAlpha = 0.55
			ctx.beginPath(); ctx.moveTo(a.X(0), a.Y(0))
			for (let l = 0; l <= n; l++) ctx.lineTo(a.X(l * dt), a.Y(Math.pow(g, l)))
			ctx.lineTo(a.X(n * dt), a.Y(0)); ctx.closePath(); ctx.fill()
			ctx.globalAlpha = 1
		}
		// horizon marker
		const tx = Math.min(tau, XMAX)
		ctx.strokeStyle = c.ink; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4])
		ctx.beginPath(); ctx.moveTo(a.X(tx), a.Y(1.05)); ctx.lineTo(a.X(tx), a.Y(0)); ctx.stroke(); ctx.setLineDash([])
		ctx.fillStyle = c.ink; ctx.font = mono
		const lab = tau > XMAX ? `τ = ${num(tau, 1)} s →` : `τ = ${num(tau, 2)} s`
		const lw = ctx.measureText(lab).width
		ctx.fillText(lab, Math.min(a.X(tx) + 6, W - 16 - lw), a.Y(0.8))
		a.draw(c, T('time ahead [s] →', 'ileri zaman [s] →'), T('weight γ^l', 'ağırlık γ^l'))
		ctx.fillStyle = c.muted
		for (let s = 0; s < XMAX; s++) ctx.fillText(String(s), a.X(s) - 4, a.Y(0) + 16)
		rTau(`${num(tau, tau < 10 ? 2 : 1)} s`)
		rSteps(num(1 / (1 - g), 0))
		rOne(pct(100 * Math.pow(g, 1 / dt), 1))
		rHalf(`${num((Math.log(0.5) / Math.log(g)) * dt, 2)} s`)
	})
}
