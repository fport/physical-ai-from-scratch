// Lesson 25 · Wilson and Wald 95% intervals for k successes in n trials (the lesson's formulas,
// z = 1.96), plus 25 simulated evaluations of n Bernoulli trials at a chosen long-run rate p. The
// coverage readouts are exact binomial sums, not samples. Default p = 0.737 is policy A's pool rate.
import { animate, colors, frame, mono, rng, type Lang } from './_kit'

const Z = 1.96
const wilson = (k: number, n: number): [number, number] => {
	const p = k / n, den = 1 + (Z * Z) / n
	const c = (p + (Z * Z) / (2 * n)) / den
	const h = (Z * Math.sqrt((p * (1 - p)) / n + (Z * Z) / (4 * n * n))) / den
	return [Math.max(0, c - h), Math.min(1, c + h)]
}
const wald = (k: number, n: number): [number, number] => {
	const p = k / n, h = Z * Math.sqrt((p * (1 - p)) / n)
	return [Math.max(0, p - h), Math.min(1, p + h)]
}
/** probability that the interval built from k ~ Binomial(n, p) contains p */
const coverage = (n: number, p: number, ci: (k: number, n: number) => [number, number]) => {
	let pmf = Math.pow(1 - p, n), cov = 0
	for (let k = 0; k <= n; k++) {
		const [lo, hi] = ci(k, n)
		if (lo <= p && p <= hi) cov += pmf
		pmf *= ((n - k) / (k + 1)) * (p / (1 - p))
	}
	return cov
}

export default function (root: HTMLElement, lang: Lang) {
	const f = frame(root, lang, {
		title: { en: 'How wide is 9 out of 10?', tr: '10’da 9 ne kadar geniş?' },
		caption: {
			en: 'Top: the 95% interval for your k/n. Bottom: 25 evaluations of n trials of a policy whose long-run rate is p; intervals that miss p are highlighted. Push k to n and watch Wald collapse to a point while Wilson stays honest.',
			tr: 'Üstte: senin k/n’in için %95 aralığı. Altta: uzun vadeli oranı p olan bir policy’nin n denemelik 25 evaluation’ı; p’yi kaçıran aralıklar vurgulu. k’yı n’ye it: Wald bir noktaya çöker, Wilson dürüst kalır.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 360)
	const num = (x: number, d: number) => (lang === 'tr' ? x.toFixed(d).replace('.', ',') : x.toFixed(d))
	const n = f.slider({ en: 'trials n', tr: 'deneme n' }, { min: 1, max: 100, step: 1, value: 10 })
	const k = f.slider({ en: 'successes k', tr: 'başarı k' }, { min: 0, max: 100, step: 1, value: 9 })
	const p = f.slider({ en: 'long-run rate p', tr: 'uzun vadeli oran p' }, { min: 0.05, max: 0.99, step: 0.001, value: 0.737, fmt: (x) => num(x, 3) })
	const rWil = f.readout({ en: 'Wilson', tr: 'Wilson' })
	const rWald = f.readout({ en: 'Wald', tr: 'Wald' })
	const rCov = f.readout({ en: 'covers p (exact)', tr: 'p’yi kapsar (exact)' })
	n.on((v) => { if (k.get() > v) k.set(v) })
	k.on((v) => { if (v > n.get()) k.set(n.get()) })

	let seed = 1, sims: number[] = []
	const simulate = () => {
		const r = rng(seed * 104729)
		sims = Array.from({ length: 25 }, () => { let s = 0; for (let i = 0; i < n.get(); i++) if (r.u() < p.get()) s++; return s })
	}
	simulate()
	f.button({ en: 'New evaluations', tr: 'Yeni evaluation’lar' }, () => { seed++; simulate() })
	n.on(simulate); p.on(simulate)

	const x0 = 40, x1 = W - 24
	const X = (v: number) => x0 + v * (x1 - x0)
	const bar = (lo: number, hi: number, y: number, color: string, w: number, dash: number[] = []) => {
		ctx.strokeStyle = color; ctx.lineWidth = w; ctx.setLineDash(dash)
		ctx.beginPath(); ctx.moveTo(X(lo), y); ctx.lineTo(X(hi), y); ctx.stroke(); ctx.setLineDash([])
		ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(lo), y - 6); ctx.lineTo(X(lo), y + 6); ctx.moveTo(X(hi), y - 6); ctx.lineTo(X(hi), y + 6); ctx.stroke()
	}

	animate(root, () => {
		const c = colors(), N = n.get(), K = Math.min(k.get(), N), P = p.get()
		ctx.clearRect(0, 0, W, H)
		ctx.font = mono
		// axis 0..1
		const axisY = 118
		ctx.strokeStyle = c.line; ctx.lineWidth = 1.5
		ctx.beginPath(); ctx.moveTo(x0, axisY); ctx.lineTo(x1, axisY); ctx.stroke()
		ctx.fillStyle = c.muted
		for (let i = 0; i <= 10; i++) {
			ctx.fillRect(X(i / 10) - 0.5, axisY - 4, 1, 8)
			if (i % 2 === 0) ctx.fillText(num(i / 10, 1), X(i / 10) - 11, axisY + 18)
		}
		// your result
		const wi = wilson(K, N), wa = wald(K, N)
		ctx.fillStyle = c.muted
		ctx.fillText(`Wilson · k/n = ${K}/${N}`, x0, 22)
		bar(wi[0], wi[1], 42, c.accent, 6)
		ctx.fillStyle = c.muted; ctx.fillText('Wald', x0, 70)
		bar(wa[0], wa[1], 88, c.ink, 3, [6, 4])
		ctx.fillStyle = c.ink; ctx.beginPath(); ctx.arc(X(K / N), 42, 4, 0, 7); ctx.fill()
		// simulated evaluations
		const top = 166, rowH = (H - top - 10) / sims.length
		ctx.fillStyle = c.muted
		ctx.fillText(lang === 'tr' ? `p = ${num(P, 3)} olan bir policy’nin 25 evaluation’ı (Wilson)` : `25 evaluations of a policy with p = ${num(P, 3)} (Wilson)`, x0, top - 12)
		ctx.strokeStyle = c.line; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4])
		ctx.beginPath(); ctx.moveTo(X(P), top - 4); ctx.lineTo(X(P), H - 6); ctx.stroke(); ctx.setLineDash([])
		let miss = 0
		sims.forEach((s, i) => {
			const [lo, hi] = wilson(s, N), y = top + (i + 0.5) * rowH
			const hit = lo <= P && P <= hi
			if (!hit) miss++
			ctx.strokeStyle = hit ? c.muted : c.accent; ctx.lineWidth = hit ? 2 : 3.5
			ctx.beginPath(); ctx.moveTo(X(lo), y); ctx.lineTo(X(hi), y); ctx.stroke()
			ctx.fillStyle = hit ? c.muted : c.accent; ctx.fillRect(X(s / N) - 2, y - 2, 4, 4)
		})
		rWil(`[${num(wi[0], 3)}, ${num(wi[1], 3)}]`)
		rWald(`[${num(wa[0], 3)}, ${num(wa[1], 3)}]`)
		rCov(`Wilson ${num(coverage(N, P, wilson), 3)} · Wald ${num(coverage(N, P, wald), 3)} · ${lang === 'tr' ? 'burada kaçıran' : 'missed here'} ${miss}/25`)
	})
}
