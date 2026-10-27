// Lesson 30 · MSE regression vs a flow-matching sampler on the lesson's task: go from (0, −1) to
// (0, +1) past a round obstacle of radius 0.30. Every path here is x(y) = c·(1 − y²), so one number c
// (the lateral offset at y = 0) describes it; demonstrations pass at c ≈ −0.55 or +0.55. MSE returns
// the mean of c. The flow sampler integrates, with N Euler steps from Gaussian noise, the exact
// straight-path flow-matching velocity of the two-mode demo distribution (no network is trained,
// so it is the best case of the method, not a learned model).
import { animate, axes, colors, frame, mono, rng, type Lang } from './_kit'

export default function (root: HTMLElement, lang: Lang) {
	const f = frame(root, lang, {
		title: { en: 'The average of two good paths goes through the obstacle', tr: 'İki iyi yolun ortalaması engelin içinden geçer' },
		caption: {
			en: 'Left: demonstrations (grey), the MSE path (dashed) and 40 sampled paths. Right: how each sample moves from noise (t = 0) to a path (t = 1) in N Euler steps. One step lands every sample on the average; more steps let each one commit to a side. The sampler follows the exact flow of the demo distribution, so it is cleaner than a trained network.',
			tr: 'Solda: gösterimler (gri), MSE yolu (kesikli) ve örneklenmiş 40 yol. Sağda: her örneğin gürültüden (t = 0) bir yola (t = 1) N Euler adımında nasıl gittiği. Tek adım her örneği ortalamaya düşürür; daha çok adım her birinin bir tarafa karar vermesini sağlar. Örnekleyici gösterim dağılımının tam akışını izliyor; bu yüzden eğitilmiş bir ağdan daha temiz.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 320)
	const STEPS = [1, 2, 4, 8, 16, 32]
	const left = f.slider({ en: 'demos passing left', tr: 'soldan geçen gösterim' }, { min: 50, max: 100, step: 5, value: 50, fmt: (x) => (lang === 'tr' ? '%' + x : x + '%') })
	const stepI = f.slider({ en: 'Euler steps N', tr: 'Euler adımı N' }, { min: 0, max: 5, step: 1, value: 3, fmt: (i) => String(STEPS[i]) })
	const rMse = f.readout({ en: 'MSE path at y = 0', tr: 'MSE yolu, y = 0’da' })
	const rHit = f.readout({ en: 'samples that hit', tr: 'çarpan örnekler' })
	const rLR = f.readout({ en: 'left / right', tr: 'sol / sağ' })
	const num = (x: number, d: number) => (lang === 'tr' ? x.toFixed(d).replace('.', ',') : x.toFixed(d))

	const MU = 0.55, SD = 0.03, RAD = 0.3, NS = 40
	const vel = (c: number, t: number, pl: number) => {
		const st2 = (1 - t) ** 2 + t * t * SD * SD
		const comps: [number, number][] = [[pl, -MU], [1 - pl, MU]]
		const logw = comps.map(([pk, mk]) => (pk > 0 ? Math.log(pk) - ((c - t * mk) ** 2) / (2 * st2) : -Infinity))
		const mx = Math.max(...logw)
		let ws = 0, e = 0
		comps.forEach(([, mk], i) => {
			const w = Math.exp(logw[i] - mx)
			ws += w; e += w * (mk + ((t * SD * SD) / st2) * (c - t * mk))
		})
		return (e / ws - c) / (1 - t)
	}
	let seed = 1, noise: number[] = [], demoC: number[] = []
	const draw0 = () => {
		const r = rng(seed * 31337)
		noise = Array.from({ length: NS }, () => r.gauss())
		const rd = rng(5)
		demoC = Array.from({ length: 20 }, (_, i) => (i < Math.round((20 * left.get()) / 100) ? -MU : MU) + SD * rd.gauss())
	}
	draw0()
	left.on(draw0)
	f.button({ en: 'New samples', tr: 'Yeni örnekler' }, () => { seed++; draw0() })

	// scene panel: x in [-1, 1], y in [-1.1, 1.1] -> square at the left
	const sx0 = 14, sy0 = 10, side = 300
	const PX = (x: number) => sx0 + ((x + 1) / 2) * side
	const PY = (y: number) => sy0 + ((1.1 - y) / 2.2) * side
	const path = (cc: number, color: string, w: number, dash: number[] = []) => {
		ctx.strokeStyle = color; ctx.lineWidth = w; ctx.setLineDash(dash); ctx.beginPath()
		for (let i = 0; i <= 40; i++) { const y = -1 + i / 20, x = cc * (1 - y * y); i ? ctx.lineTo(PX(x), PY(y)) : ctx.moveTo(PX(x), PY(y)) }
		ctx.stroke(); ctx.setLineDash([])
	}

	animate(root, () => {
		const c = colors(), pl = left.get() / 100, N = STEPS[stepI.get()]
		const mean = pl * -MU + (1 - pl) * MU
		// integrate every sample, keeping its trajectory for the right panel
		const trajs = noise.map((c0) => {
			const tr: [number, number][] = [[0, c0]]
			let v = c0
			for (let i = 0; i < N; i++) { v += vel(v, i / N, pl) / N; tr.push([(i + 1) / N, v]) }
			return tr
		})
		const finals = trajs.map((tr) => tr[tr.length - 1][1])
		ctx.clearRect(0, 0, W, H)
		// scene
		ctx.fillStyle = c.line; ctx.beginPath(); ctx.arc(PX(0), PY(0), (RAD / 2) * side, 0, 7); ctx.fill()
		for (const d of demoC) path(d, c.muted, 1)
		finals.forEach((fc) => path(fc, Math.abs(fc) < RAD ? c.ink : c.accent, Math.abs(fc) < RAD ? 2 : 1.5))
		path(mean, c.ink, 3, [7, 5])
		ctx.fillStyle = c.ink
		for (const y of [-1, 1]) { ctx.beginPath(); ctx.arc(PX(0), PY(y), 5, 0, 7); ctx.fill() }
		ctx.fillStyle = c.muted; ctx.font = mono
		ctx.fillText(lang === 'tr' ? 'engel' : 'obstacle', PX(0) - 26, PY(0) + 4)
		ctx.fillText(lang === 'tr' ? 'başlangıç' : 'start', PX(0) + 10, PY(-1) + 4)
		ctx.fillText(lang === 'tr' ? 'hedef' : 'goal', PX(0) + 10, PY(1) + 4)
		// sampling panel: c over t
		const a = axes(ctx, { x0: 360, y0: 16, x1: W - 14, y1: H - 30 }, [0, 1], [-1.8, 1.8])
		ctx.fillStyle = c.panel; ctx.fillRect(a.X(0), a.Y(RAD), a.X(1) - a.X(0), a.Y(-RAD) - a.Y(RAD))
		a.draw(c, lang === 'tr' ? 't (gürültü → yol) →' : 't (noise → path) →', lang === 'tr' ? 'y = 0’daki ofset c' : 'offset c at y = 0')
		ctx.fillStyle = c.muted; ctx.fillText(lang === 'tr' ? 'engel bandı' : 'obstacle band', a.X(0) + 6, a.Y(0) + 4)
		for (const m of [-MU, MU]) a.line([[0, m], [1, m]], c.line, 1, [4, 4])
		ctx.save(); ctx.beginPath(); ctx.rect(a.X(0), a.Y(1.8), a.X(1) - a.X(0), a.Y(-1.8) - a.Y(1.8)); ctx.clip()
		trajs.forEach((tr, i) => a.line(tr, Math.abs(finals[i]) < RAD ? c.ink : c.accent, 1.2))
		ctx.restore()
		const hit = finals.filter((v) => Math.abs(v) < RAD).length
		const nl = finals.filter((v) => v <= -RAD).length
		rMse(`c = ${num(mean, 2)} ${Math.abs(mean) < RAD ? (lang === 'tr' ? '(engelin içinde)' : '(inside the obstacle)') : ''}`)
		rHit(`${hit}/${NS}`)
		rLR(`${nl} / ${NS - hit - nl}`)
	})
}
