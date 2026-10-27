// Lesson 22 · a single-tree RRT with goal bias in a 2-D configuration space. The planner never sees
// the obstacles; it only asks free(q) at points spaced `res` apart along each new edge, exactly like
// edge_free() in the lesson. A coarse res lets an edge jump through the thin wall or clip a corner;
// the dense re-check (0.005 rad) of the found path shows it. The res values are the lesson's sweep.
import { animate, colors, frame, mono, rng, type Lang } from './_kit'

type Q = [number, number]

export default function (root: HTMLElement, lang: Lang) {
	const f = frame(root, lang, {
		title: { en: 'RRT that only sees its collision checker', tr: 'Yalnızca collision checker’ını gören RRT' },
		caption: {
			en: 'A 2-joint C-space with a block and a thin wall. Each new edge is tested only at points spaced “edge check” apart. Make the check coarse and the tree reports a path that passes through the wall; the dense re-check marks those segments.',
			tr: 'Bir blok ve ince bir duvarın olduğu 2 joint’lik bir C-space. Her yeni edge yalnızca “edge kontrolü” aralıklı noktalarda test ediliyor. Kontrolü kabalaştır: ağaç duvarın içinden geçen bir path bildirir; yoğun yeniden kontrol o parçaları işaretler.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 320)
	const RES = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
	const eps = f.slider({ en: 'step ε [rad]', tr: 'adım ε [rad]' }, { min: 0.1, max: 1, step: 0.05, value: 0.4, fmt: (x) => x.toFixed(2) })
	const resI = f.slider({ en: 'edge check every [rad]', tr: 'edge kontrolü her [rad]' }, { min: 0, max: 5, step: 1, value: 1, fmt: (i) => RES[i].toFixed(2) })
	const bias = f.slider({ en: 'goal bias', tr: 'goal bias' }, { min: 0, max: 0.5, step: 0.05, value: 0.05, fmt: (x) => Math.round(x * 100) + '%' })
	const rIt = f.readout({ en: 'iterations', tr: 'iterasyon' })
	const rChk = f.readout({ en: 'collision checks', tr: 'collision check' })
	const rPath = f.readout({ en: 'dense re-check', tr: 'yoğun yeniden kontrol' })

	// C-space box and obstacles (rectangles in rad: q1 from, q2 from, q1 to, q2 to)
	const Q1: Q = [-3, 3], Q2: Q = [-1.5, 1.5]
	const obstacles: [number, number, number, number][] = [
		[-0.9, -0.6, 0.0, 1.5], // block hanging from the top: go below it
		[1.2, -1.5, 1.28, 0.9], // thin wall standing on the bottom: go over it
		[-2.1, -1.5, -1.6, -0.9], // small block near the start
	]
	const start: Q = [-2.6, 0], goal: Q = [2.6, 0]
	const free = (q: Q) => !obstacles.some(([a, b, c, d]) => q[0] >= a && q[0] <= c && q[1] >= b && q[1] <= d)
	const dist = (a: Q, b: Q) => Math.hypot(b[0] - a[0], b[1] - a[1])
	const lerp = (a: Q, b: Q, s: number): Q => [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s]

	let checks = 0
	const edgeFree = (a: Q, b: Q, res: number, count = true) => {
		const n = Math.max(1, Math.ceil(dist(a, b) / res))
		for (let k = 1; k <= n; k++) {
			if (count) checks++
			if (!free(lerp(a, b, k / n))) return false
		}
		return true
	}

	let seed = 1, R = rng(seed)
	let nodes: Q[] = [], parent: number[] = [], it = 0
	let path: Q[] | null = null, badSeg: boolean[] = [], done = false
	const BUDGET = 3000
	const reset = () => {
		R = rng(seed * 7919)
		nodes = [start]; parent = [-1]; it = 0; checks = 0
		path = null; badSeg = []; done = false
	}
	reset()
	f.button({ en: 'New seed', tr: 'Yeni seed' }, () => { seed++; reset() })
	for (const c of [eps, resI, bias]) c.on(reset)

	const iterate = () => {
		const res = RES[resI.get()], e = eps.get()
		it++
		const qr: Q = R.u() < bias.get() ? goal : [Q1[0] + R.u() * (Q1[1] - Q1[0]), Q2[0] + R.u() * (Q2[1] - Q2[0])]
		let best = 0, bd = Infinity
		for (let i = 0; i < nodes.length; i++) { const d = dist(nodes[i], qr); if (d < bd) { bd = d; best = i } }
		const qn = nodes[best]
		const qnew = bd <= e ? qr : lerp(qn, qr, e / bd)
		if (!edgeFree(qn, qnew, res)) return
		nodes.push(qnew); parent.push(best)
		if (dist(qnew, goal) <= e && edgeFree(qnew, goal, res)) {
			nodes.push(goal); parent.push(nodes.length - 2)
			const p: Q[] = []
			for (let i = nodes.length - 1; i >= 0; i = parent[i]) p.push(nodes[i])
			path = p.reverse()
			badSeg = path.slice(1).map((q, i) => !edgeFree(path![i], q, 0.005, false)) // ground truth
			done = true
		}
	}

	const X = (q: number) => 20 + ((q - Q1[0]) / (Q1[1] - Q1[0])) * (W - 40)
	const Y = (q: number) => 10 + ((Q2[1] - q) / (Q2[1] - Q2[0])) * (H - 20)

	animate(root, () => {
		for (let i = 0; i < 4 && !done && it < BUDGET; i++) iterate()
		if (!done && it >= BUDGET) done = true
		const c = colors()
		ctx.clearRect(0, 0, W, H)
		ctx.strokeStyle = c.line; ctx.lineWidth = 1.5
		ctx.strokeRect(X(Q1[0]), Y(Q2[1]), X(Q1[1]) - X(Q1[0]), Y(Q2[0]) - Y(Q2[1]))
		ctx.fillStyle = c.line
		for (const [a, b, cc, d] of obstacles) ctx.fillRect(X(a), Y(d), X(cc) - X(a), Y(b) - Y(d))
		// tree
		ctx.strokeStyle = c.muted; ctx.lineWidth = 1
		ctx.beginPath()
		for (let i = 1; i < nodes.length; i++) {
			const a = nodes[parent[i]], b = nodes[i]
			ctx.moveTo(X(a[0]), Y(a[1])); ctx.lineTo(X(b[0]), Y(b[1]))
		}
		ctx.stroke()
		// path, with the points the checker actually tested
		if (path) {
			const res = RES[resI.get()]
			for (let i = 1; i < path.length; i++) {
				const a = path[i - 1], b = path[i]
				ctx.strokeStyle = badSeg[i - 1] ? c.accent : c.ink
				ctx.lineWidth = badSeg[i - 1] ? 4 : 2.5
				ctx.beginPath(); ctx.moveTo(X(a[0]), Y(a[1])); ctx.lineTo(X(b[0]), Y(b[1])); ctx.stroke()
				const n = Math.max(1, Math.ceil(dist(a, b) / res))
				ctx.fillStyle = c.ink
				for (let k = 1; k <= n; k++) { const q = lerp(a, b, k / n); ctx.fillRect(X(q[0]) - 1.5, Y(q[1]) - 1.5, 3, 3) }
			}
		}
		ctx.fillStyle = c.ink; ctx.beginPath(); ctx.arc(X(start[0]), Y(start[1]), 6, 0, 7); ctx.fill()
		ctx.strokeStyle = c.ink; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(X(goal[0]), Y(goal[1]), 7, 0, 7); ctx.stroke()
		ctx.fillStyle = c.muted; ctx.font = mono
		ctx.fillText(lang === 'tr' ? 'başlangıç' : 'start', X(start[0]) - 26, Y(start[1]) + 22)
		ctx.fillText(lang === 'tr' ? 'hedef' : 'goal', X(goal[0]) - 18, Y(goal[1]) + 24)
		ctx.fillText('q1 →', W - 62, H - 16)
		ctx.fillText('q2 ↑', 26, 28)
		rIt(String(it))
		rChk(String(checks))
		const nBad = badSeg.filter(Boolean).length
		rPath(!path ? (done ? (lang === 'tr' ? 'bütçe bitti' : 'budget used up') : (lang === 'tr' ? 'arıyor…' : 'searching…'))
			: nBad ? (lang === 'tr' ? `${nBad} parça engelin içinden` : `${nBad} segment(s) through an obstacle`)
			: (lang === 'tr' ? 'path boş' : 'path is free'))
	})
}
