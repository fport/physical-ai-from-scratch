// Lesson 35 · why "end the episode on success" plus a positive per-step reward teaches the arm to
// hover. The gripper is 3 cm from the target with N policy steps left. It can enter the 2 cm success
// ball (to 1 cm) or park at a distance d and stay there. Discounted value with the lesson's
// γ = 0.95: parking is r(d)·(1 − γ^N)/(1 − γ); entering is r(1 cm), and nothing after it if the
// episode terminates. Rewards are the lesson's closeness 1 − tanh(d / 5 cm) and dense −d.
import { animate, axes, colors, frame, mono, type Lang } from './_kit'

export default function (root: HTMLElement, lang: Lang) {
	const T = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => (lang === 'tr' ? x.toFixed(d).replace('.', ',') : x.toFixed(d))
	const f = frame(root, lang, {
		title: { en: 'Enter the ball, or hover just outside?', tr: 'Topun içine mi girsin, hemen dışında mı beklesin?' },
		caption: {
			en: 'The arm is 3 cm from the target. Left: what an agent that maximizes the discounted return does. Right: the return of parking at each distance d, and of entering the success ball. Flip the sign of the reward or switch termination off and the best choice changes.',
			tr: 'Kol hedefe 3 cm uzakta. Solda: discounted return’ü maksimize eden bir agent’ın yaptığı. Sağda: her d mesafesinde park etmenin ve başarı topuna girmenin return’ü. Reward’un işaretini çevir ya da termination’ı kapat; en iyi seçim değişir.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 290)
	const reward = f.select({ en: 'reward', tr: 'reward' }, [
		{ en: 'closeness 1 − tanh(d/5 cm)  (> 0)', tr: 'yakınlık 1 − tanh(d/5 cm)  (> 0)' },
		{ en: 'dense −d  (< 0)', tr: 'dense −d  (< 0)' },
	])
	const term = f.checkbox({ en: 'episode ends on success (d < 2 cm)', tr: 'başarıda episode biter (d < 2 cm)' }, true)
	const steps = f.slider({ en: 'steps left N', tr: 'kalan adım N' }, { min: 5, max: 80, step: 1, value: 60 })
	const rEnter = f.readout({ en: 'return if it enters', tr: 'girerse return' })
	const rPark = f.readout({ en: 'best hover', tr: 'en iyi bekleme' })
	const rPick = f.readout({ en: 'the optimum', tr: 'optimum' })

	const G = 0.95, BALL = 0.02, D_IN = 0.01, D0 = 0.03
	const r = (d: number) => (reward.get() === 0 ? 1 - Math.tanh(d / 0.05) : -d)
	const sumG = (n: number) => (1 - Math.pow(G, n)) / (1 - G)
	// value of moving to distance d now and staying there for the remaining steps
	const value = (d: number, n: number) => (d < BALL && term.get() ? r(d) : r(d) * sumG(n))

	// best hover distance outside the ball (grid search over 2..10 cm)
	const best = (n: number) => {
		let bd = BALL, bv = -Infinity
		for (let i = 0; i <= 800; i++) { const d = BALL + (i / 800) * 0.08; const v = value(d, n); if (v > bv) { bv = v; bd = d } }
		return { d: bd, v: bv }
	}

	// animation state: the optimal agent from 3 cm, one policy step per 40 ms
	let k = 0, acc = 0, timer = 0, pause = 0
	const restart = () => { k = 0; acc = 0; timer = 0; pause = 0 }
	reward.on(restart); term.on(restart); steps.on(restart)

	animate(root, (dt) => {
		const c = colors()
		const n = steps.get()
		const vin = value(D_IN, n), bh = best(n)
		const enters = vin >= bh.v
		const goal = enters ? D_IN : bh.d
		const ends = enters && term.get()
		const total = ends ? 1 : n
		// advance the episode
		if (pause > 0) { pause -= dt; if (pause <= 0) restart() }
		else {
			timer += dt
			while (timer >= 0.04 && k < total) { acc += Math.pow(G, k) * r(goal); k++; timer -= 0.04 }
			if (k >= total) pause = 1.6
		}
		ctx.clearRect(0, 0, W, H)
		// left: top view, 1 cm = 10 px
		const cx = 112, cy = 132, S = 1000
		ctx.strokeStyle = c.line; ctx.lineWidth = 1; ctx.setLineDash([3, 4])
		ctx.beginPath(); ctx.arc(cx, cy, 0.05 * S, 0, 7); ctx.stroke()
		ctx.beginPath(); ctx.arc(cx, cy, 0.1 * S, 0, 7); ctx.stroke(); ctx.setLineDash([])
		ctx.fillStyle = c.soft; ctx.strokeStyle = c.accent; ctx.lineWidth = 2
		ctx.beginPath(); ctx.arc(cx, cy, BALL * S, 0, 7); ctx.fill(); ctx.stroke()
		ctx.fillStyle = c.accent; ctx.beginPath(); ctx.arc(cx, cy, 3, 0, 7); ctx.fill()
		ctx.fillStyle = c.muted; ctx.font = mono
		ctx.fillText('5 cm', cx + 0.05 * S * 0.71 + 2, cy - 0.05 * S * 0.71 - 2)
		ctx.fillText('10 cm', cx + 0.1 * S * 0.71 + 2, cy - 0.1 * S * 0.71 - 2)
		// gripper: approaches from 3 cm along a ray, with a small wobble while hovering
		const ang = -0.6
		const frac = Math.min(1, k + timer / 0.04)
		const dnow = D0 + (goal - D0) * frac
		const wob = !ends && k > 1 ? 0.0012 * Math.sin(8 * (k + timer / 0.04)) : 0
		const gx = cx + (dnow + wob) * S * Math.cos(ang), gy = cy + (dnow + wob) * S * Math.sin(ang)
		const over = ends && k >= 1
		ctx.globalAlpha = over ? 0.35 : 1
		ctx.fillStyle = c.ink; ctx.beginPath(); ctx.arc(gx, gy, 6, 0, 7); ctx.fill()
		ctx.globalAlpha = 1
		ctx.fillStyle = c.ink
		ctx.fillText(T(`step ${k}/${total}`, `adım ${k}/${total}`), 16, 268)
		ctx.fillText(T(`return ${num(acc, 2)}`, `return ${num(acc, 2)}`), 16, 284)
		if (over) ctx.fillText(T('episode over', 'episode bitti'), 16, 22)
		else if (!enters) ctx.fillText(T(`hovers at ${num(bh.d * 100, 1)} cm`, `${num(bh.d * 100, 1)} cm’de bekliyor`), 16, 22)
		else ctx.fillText(T('stays inside the ball', 'topun içinde kalıyor'), 16, 22)

		// right: value of each choice against the parking distance
		const pts: [number, number][] = [], inside: [number, number][] = []
		let lo = Math.min(0, vin), hi = Math.max(0, vin)
		for (let i = 0; i <= 200; i++) {
			const d = (i / 200) * 0.1, v = value(d, n)
			lo = Math.min(lo, v); hi = Math.max(hi, v)
			if (d >= BALL) pts.push([d * 100, v]); else inside.push([d * 100, v])
		}
		const pad = 0.08 * (hi - lo || 1)
		const a = axes(ctx, { x0: 300, y0: 34, x1: W - 16, y1: H - 34 }, [0, 10], [lo - pad, hi + pad])
		ctx.fillStyle = c.soft; ctx.fillRect(a.X(0), a.Y(hi + pad), a.X(2) - a.X(0), a.Y(lo - pad) - a.Y(hi + pad))
		a.line([[0, 0], [10, 0]], c.line, 1, [2, 4])
		a.line(inside, c.muted, 2, [4, 3])
		a.line(pts, c.accent)
		// markers: enter (1 cm) and best hover
		ctx.fillStyle = c.ink; ctx.beginPath(); ctx.arc(a.X(1), a.Y(vin), 5, 0, 7); ctx.fill()
		ctx.strokeStyle = c.accent; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(a.X(bh.d * 100), a.Y(bh.v), 6, 0, 7); ctx.stroke()
		a.draw(c, T('parking distance d [cm] →', 'park mesafesi d [cm] →'), T('discounted return', 'discounted return'))
		ctx.fillStyle = c.muted; ctx.font = mono
		for (const s of [0, 2, 4, 6, 8]) ctx.fillText(String(s), a.X(s) - 4, a.Y(lo - pad) + 16)
		ctx.fillText(T('ball', 'top'), a.X(0) + 4, a.Y(lo - pad) - 6)
		ctx.fillStyle = c.ink
		ctx.fillText(T('enter', 'gir'), a.X(1) + 9, a.Y(vin) + (vin > (lo + hi) / 2 ? 16 : -8))

		rEnter(num(vin, 2))
		rPark(`${num(bh.d * 100, 1)} cm → ${num(bh.v, 2)}`)
		rPick(enters ? T('enter', 'gir') : T('hover outside', 'dışarıda bekle'))
	})
}
