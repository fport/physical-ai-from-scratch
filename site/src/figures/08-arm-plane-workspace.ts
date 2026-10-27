// Lesson 08 · the SO-101's three parallel pitch joints (shoulder_lift, elbow_flex, wrist_flex) seen in
// the arm plane, with pan = 0 and wrist_roll = 0. Joint positions at q = 0 and the joint ranges are read
// from the Menagerie model the lesson uses; FK is the planar product of rotations about the joint axes.
// The sampling buttons draw joint vectors uniformly within the ranges and count occupied cells.
import { animate, axes, colors, frame, mono, type Lang } from './_kit'

// arm plane coordinates (x forward, z up) in metres at q = 0: lift axis, elbow axis, wrist axis, tool site
const O1 = [0.06923, 0.1166], O2 = [0.09723, 0.22917], O3 = [0.23213, 0.23437], TIP = [0.39136, 0.24635]
const LIM = [100, 96.8, 95].map((d) => (d * Math.PI) / 180) // ± range of lift, elbow, wrist flex
const D = (Math.PI / 180)

/** rotation about the joint axis (+y in the model): positive angles turn the arm downwards */
const rot = (v: number[], q: number) => [v[0] * Math.cos(q) + v[1] * Math.sin(q), -v[0] * Math.sin(q) + v[1] * Math.cos(q)]
const add = (a: number[], b: number[]) => [a[0] + b[0], a[1] + b[1]]
const sub = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1]]

function fk(q: number[]) {
	const a3 = q[0] + q[1] + q[2], a2 = q[0] + q[1]
	const p2 = add(O1, rot(sub(O2, O1), q[0]))
	const p3 = add(p2, rot(sub(O3, O2), a2))
	const pt = add(p3, rot(sub(TIP, O3), a3))
	return { p2, p3, pt, approach: rot([1, 0], a3) }
}

export default function (root: HTMLElement, lang: Lang) {
	const L = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => x.toFixed(d).replace('.', L('.', ','))
	const int = (x: number) => Math.round(x).toLocaleString(lang === 'tr' ? 'tr-TR' : 'en-US')
	const f = frame(root, lang, {
		title: { en: 'Workspace by sampling', tr: 'Örnekleyerek workspace' },
		caption: {
			en: 'The SO-101’s three pitch joints in the arm plane (pan turns this plane, wrist roll spins about the approach axis). Move the joints, then sample random joint vectors within the ranges. Watch the reach stop growing early while the count of 1 cm cells keeps climbing: uniform in joint space is not uniform in space.',
			tr: 'SO-101’in kol düzlemindeki üç pitch joint’i (pan bu düzlemi döndürür, wrist roll yaklaşma ekseni etrafında döner). Joint’leri oynat, sonra aralıklar içinde rastgele joint vektörleri örnekle. Erişimin erkenden büyümeyi bıraktığını, 1 cm’lik hücre sayısının ise tırmanmaya devam ettiğini izle: joint uzayında düzgün olmak uzayda düzgün olmak değildir.',
		},
	})
	const { ctx, W, H, cv } = f.canvas(640, 360)
	const deg = (x: number) => `${num(x, 0)}°`
	const q1 = f.slider({ en: 'shoulder_lift', tr: 'shoulder_lift' }, { min: -100, max: 100, step: 1, value: -30, fmt: deg })
	const q2 = f.slider({ en: 'elbow_flex', tr: 'elbow_flex' }, { min: -96.8, max: 96.8, step: 0.1, value: 40, fmt: deg })
	const q3 = f.slider({ en: 'wrist_flex', tr: 'wrist_flex' }, { min: -95, max: 95, step: 1, value: 30, fmt: deg })
	const rTip = f.readout({ en: 'tool (x, z)', tr: 'alet (x, z)' })
	const rReach = f.readout({ en: 'max reach so far', tr: 'şimdiye kadar en büyük erişim' })
	const rN = f.readout({ en: 'samples', tr: 'örnek' })
	const rA1 = f.readout({ en: 'area, 1 cm cells', tr: 'alan, 1 cm hücre' })
	const rA2 = f.readout({ en: 'area, 2 cm cells', tr: 'alan, 2 cm hücre' })
	const rBelow = f.readout({ en: 'tool below z = 0', tr: 'alet z = 0’ın altında' })

	// drawing frame for the arm plane
	const S = 400, ox = 160, oy = 240
	const X = (x: number) => ox + x * S, Y = (z: number) => oy - z * S
	// sampled points accumulate on an offscreen canvas at the same pixel size
	const dpr = cv.width / W
	const off = document.createElement('canvas')
	off.width = cv.width; off.height = cv.height
	const octx = off.getContext('2d')!
	octx.scale(dpr, dpr)
	const cells1 = new Set<number>(), cells2 = new Set<number>()
	let n = 0, below = 0, reach = 0, seed = 12345
	let hist: { n: number; a1: number; a2: number }[] = []
	const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647

	const sample = (count: number) => {
		const col = colors().accent
		octx.fillStyle = col; octx.globalAlpha = 0.35
		for (let i = 0; i < count; i++) {
			const q = LIM.map((l) => (2 * rand() - 1) * l)
			const p = fk(q).pt
			octx.fillRect(X(p[0]) - 0.5, Y(p[1]) - 0.5, 1, 1)
			const i1 = Math.floor(p[0] * 100) * 1000 + Math.floor(p[1] * 100)
			const i2 = Math.floor(p[0] * 50) * 1000 + Math.floor(p[1] * 50)
			cells1.add(i1); cells2.add(i2)
			if (p[1] < 0) below++
			reach = Math.max(reach, Math.hypot(p[0] - O1[0], p[1] - O1[1]))
		}
		octx.globalAlpha = 1
		n += count
		hist.push({ n, a1: cells1.size, a2: cells2.size * 4 })
	}
	const clear = () => { octx.clearRect(0, 0, W, H); cells1.clear(); cells2.clear(); n = 0; below = 0; reach = 0; seed = 12345; hist = [] }
	f.button({ en: '+1,000 samples', tr: '+1.000 örnek' }, () => sample(1000))
	f.button({ en: '+10,000', tr: '+10.000' }, () => sample(10000))
	f.button({ en: '+100,000', tr: '+100.000' }, () => { if (n < 1e6) sample(100000) })
	f.button({ en: 'Clear', tr: 'Temizle' }, clear)

	animate(root, () => {
		const c = colors()
		ctx.clearRect(0, 0, W, H)
		ctx.font = mono
		// table plane z = 0 and the base
		ctx.strokeStyle = c.line; ctx.lineWidth = 1.5
		ctx.beginPath(); ctx.moveTo(10, Y(0)); ctx.lineTo(390, Y(0)); ctx.stroke()
		ctx.fillStyle = c.muted; ctx.fillText('z = 0', 14, Y(0) - 6)
		ctx.fillStyle = c.line; ctx.fillRect(X(0) - 6, Y(O1[1]), 50, O1[1] * S)
		ctx.drawImage(off, 0, 0, W, H)
		// the arm
		const q = [q1.get() * D, q2.get() * D, q3.get() * D]
		const k = fk(q)
		const pts = [O1, k.p2, k.p3, k.pt]
		ctx.strokeStyle = c.ink; ctx.lineWidth = 8; ctx.lineCap = 'round'; ctx.lineJoin = 'round'
		ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])))); ctx.stroke()
		for (const p of pts.slice(0, 3)) { ctx.fillStyle = c.bg; ctx.strokeStyle = c.ink; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 6, 0, 7); ctx.fill(); ctx.stroke() }
		const tx = X(k.pt[0]), ty = Y(k.pt[1])
		ctx.strokeStyle = c.accent; ctx.lineWidth = 2.5
		ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(tx + 34 * k.approach[0], ty - 34 * k.approach[1]); ctx.stroke()
		ctx.fillStyle = c.accent; ctx.beginPath(); ctx.arc(tx, ty, 5, 0, 7); ctx.fill()
		ctx.fillStyle = c.muted
		ctx.fillText(L('tool + approach axis', 'alet + yaklaşma ekseni'), 14, 20)

		// right: cell area against sample count
		const a = axes(ctx, { x0: 432, y0: 40, x1: W - 14, y1: H - 40 }, [3, 6], [0, 4800])
		a.draw(c, L('samples →', 'örnek →'), L('area [cm²]', 'alan [cm²]'))
		for (const [e, lab] of [[3, '1k'], [4, '10k'], [5, '100k'], [6, '1M']] as [number, string][]) {
			ctx.fillStyle = c.muted; ctx.fillText(lab, a.X(e) - 10, H - 24)
		}
		if (hist.length) {
			a.line(hist.map((h) => [Math.log10(h.n), h.a2] as [number, number]), c.muted, 2, [5, 4])
			a.line(hist.map((h) => [Math.log10(h.n), h.a1] as [number, number]), c.accent, 2.5)
		}
		ctx.fillStyle = c.muted
		ctx.fillText(L('solid 1 cm · dashed 2 cm', 'düz 1 cm · kesikli 2 cm'), 432, 24)

		rTip(`${num(k.pt[0], 3)}, ${num(k.pt[1], 3)} m`)
		rReach(n ? `${num(reach, 3)} m` : '—')
		rN(int(n))
		rA1(n ? `${int(cells1.size)} cm²` : '—')
		rA2(n ? `${int(cells2.size * 4)} cm²` : '—')
		rBelow(n ? `${num((100 * below) / n, 1)}%`.replace(/^(.*)%$/, L('$1%', '%$1')) : '—')
	})
}
