// Lesson 10 · position IK for the SO-101's three pitch joints in the arm plane (geometry and joint
// ranges from the Menagerie model, pan = 0, wrist_roll = 0). Each solver iteration is the lesson's
// loop: e = target − x(q), Δq from the pseudo-inverse, damped least squares or the Jacobian transpose,
// then q ← clip(q + Δq) to the joint limits. Drag the target; the solver runs live.
import { animate, colors, frame, mono, type Lang } from './_kit'

const O1 = [0.06923, 0.1166], O2 = [0.09723, 0.22917], O3 = [0.23213, 0.23437], TIP = [0.39136, 0.24635]
const LIM = [100, 96.8, 95].map((d) => (d * Math.PI) / 180)
const LAMS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3] // m, the lesson's sweep
const REACH = Math.hypot(O2[0] - O1[0], O2[1] - O1[1]) + Math.hypot(O3[0] - O2[0], O3[1] - O2[1]) + Math.hypot(TIP[0] - O3[0], TIP[1] - O3[1])

const rot = (v: number[], q: number) => [v[0] * Math.cos(q) + v[1] * Math.sin(q), -v[0] * Math.sin(q) + v[1] * Math.cos(q)]
const add = (a: number[], b: number[]) => [a[0] + b[0], a[1] + b[1]]
const sub = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1]]

function fk(q: number[]) {
	const p2 = add(O1, rot(sub(O2, O1), q[0]))
	const p3 = add(p2, rot(sub(O3, O2), q[0] + q[1]))
	const pt = add(p3, rot(sub(TIP, O3), q[0] + q[1] + q[2]))
	return [O1, p2, p3, pt]
}
const jac = (pts: number[][]) => [0, 1, 2].map((i) => [pts[3][1] - pts[i][1], -(pts[3][0] - pts[i][0])]) // columns

export default function (root: HTMLElement, lang: Lang) {
	const L = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => x.toFixed(d).replace('.', L('.', ','))
	const f = frame(root, lang, {
		title: { en: 'Drag the target', tr: 'Hedefi sürükle' },
		caption: {
			en: 'Drag the target (mouse or finger); the solver chases it one iteration at a time, and the faint arm is the pose before the last step. Pull the target out of reach or near the straight arm: the pseudo-inverse takes huge steps and slams into joint limits, while damped least squares (DLS) slows down in the direction the arm cannot serve. Try λ from tiny to large.',
			tr: 'Hedefi sürükle (fare ya da parmak); çözücü onu her seferinde bir iterasyonla kovalıyor, soluk kol son adımdan önceki poz. Hedefi erişim dışına ya da dümdüz kolun yakınına çek: pseudo-inverse dev adımlar atıp joint limitlerine çarpıyor, damped least squares (DLS) ise kolun hizmet edemediği yönde yavaşlıyor. λ’yı çok küçükten büyüğe dene.',
		},
	})
	const { ctx, W, H, cv } = f.canvas(640, 340)
	const method = f.select({ en: 'method', tr: 'yöntem' }, [
		{ en: 'DLS', tr: 'DLS' },
		{ en: 'pseudo-inverse (λ = 0)', tr: 'pseudo-inverse (λ = 0)' },
		{ en: 'Jacobian transpose', tr: 'Jacobian transpose' },
	])
	const lam = f.slider({ en: 'DLS damping λ [m]', tr: 'DLS damping λ [m]' }, { min: 0, max: LAMS.length - 1, step: 1, value: 4, fmt: (i) => num(LAMS[i], 3) })
	const speed = f.select({ en: 'iterations', tr: 'iterasyon' }, [{ en: '6 per second', tr: 'saniyede 6' }, { en: '60 per second', tr: 'saniyede 60' }])
	const clamp = f.checkbox({ en: 'clip to joint limits', tr: 'joint limitlerine kırp' }, true)
	const rErr = f.readout({ en: 'error', tr: 'hata' })
	const rIt = f.readout({ en: 'iterations', tr: 'iterasyon' })
	const rDq = f.readout({ en: 'last ‖Δq‖', tr: 'son ‖Δq‖' })
	const rSig = f.readout({ en: 'σ_min', tr: 'σ_min' })
	const rLim = f.readout({ en: 'joints at a limit', tr: 'limitteki joint' })

	const S = 420, ox = 150, oy = 250
	const X = (x: number) => ox + x * S, Y = (z: number) => oy - z * S
	let q = [-0.6, 0.9, 0.4], prev = [...q], target = [0.3, 0.05], iters = 0, lastDq = 0, acc = 0
	const trail: number[][] = []
	const retarget = (p: number[]) => { target = p; iters = 0 }
	f.button({ en: 'Target out of reach', tr: 'Hedef erişim dışında' }, () => retarget([0.56, 0.2]))
	f.button({ en: 'Reset arm', tr: 'Kolu sıfırla' }, () => { q = [-0.6, 0.9, 0.4]; prev = [...q]; iters = 0; trail.length = 0 })
	for (const c of [method, speed]) c.on(() => (iters = 0))
	lam.on(() => (iters = 0))

	// pointer: drag the target anywhere on the canvas
	let dragging = false
	const toWorld = (e: PointerEvent) => {
		const r = cv.getBoundingClientRect()
		const px = ((e.clientX - r.left) * W) / r.width, py = ((e.clientY - r.top) * H) / r.height
		return [(px - ox) / S, (oy - py) / S]
	}
	cv.addEventListener('pointerdown', (e) => { dragging = true; cv.setPointerCapture(e.pointerId); retarget(toWorld(e)) })
	cv.addEventListener('pointermove', (e) => { if (dragging) retarget(toWorld(e)) })
	cv.addEventListener('pointerup', () => (dragging = false))
	cv.addEventListener('pointercancel', () => (dragging = false))
	cv.style.cursor = 'crosshair'

	const step = () => {
		const pts = fk(q), e = sub(target, pts[3])
		if (Math.hypot(e[0], e[1]) < 5e-4) return // 0.5 mm: done, as in the lesson
		const J = jac(pts) // columns j_i (2-vectors)
		let a = 0, b = 0, d = 0 // J Jᵀ
		for (const c of J) { a += c[0] * c[0]; b += c[0] * c[1]; d += c[1] * c[1] }
		let dq: number[]
		if (method.get() === 2) { // transpose with the step length from the lesson's table
			const g = [a * e[0] + b * e[1], b * e[0] + d * e[1]]
			const al = (e[0] * g[0] + e[1] * g[1]) / (g[0] * g[0] + g[1] * g[1] || 1)
			dq = J.map((c) => al * (c[0] * e[0] + c[1] * e[1]))
		} else {
			const l2 = method.get() === 0 ? LAMS[lam.get()] ** 2 : 0
			const A = [a + l2, b, b, d + l2], det = A[0] * A[3] - A[1] * A[2]
			const sdet = Math.abs(det) < 1e-18 ? (det < 0 ? -1e-18 : 1e-18) : det
			const y = [(A[3] * e[0] - A[1] * e[1]) / sdet, (-A[2] * e[0] + A[0] * e[1]) / sdet] // (J Jᵀ + λ²I)⁻¹ e
			dq = J.map((c) => c[0] * y[0] + c[1] * y[1])
		}
		prev = [...q]
		q = q.map((qi, i) => (clamp.get() ? Math.max(-LIM[i], Math.min(LIM[i], qi + dq[i])) : qi + dq[i]))
		lastDq = Math.hypot(...dq)
		iters++
		trail.push(fk(q)[3]); if (trail.length > 60) trail.shift()
	}

	animate(root, (dt) => {
		const rate = speed.get() ? 60 : 6
		acc += dt * rate
		while (acc >= 1) { step(); acc -= 1 }
		const c = colors()
		ctx.clearRect(0, 0, W, H)
		ctx.font = mono
		// reach circle around the lift axis (sum of the link lengths, ignoring limits)
		ctx.strokeStyle = c.line; ctx.lineWidth = 1; ctx.setLineDash([3, 5])
		ctx.beginPath(); ctx.arc(X(O1[0]), Y(O1[1]), REACH * S, 0, 7); ctx.stroke(); ctx.setLineDash([])
		ctx.strokeStyle = c.line; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(10, Y(0)); ctx.lineTo(W - 10, Y(0)); ctx.stroke()
		ctx.fillStyle = c.line; ctx.fillRect(X(0) - 6, Y(O1[1]), 50, O1[1] * S)
		const drawArm = (qq: number[], color: string, width: number, alpha: number) => {
			const pts = fk(qq)
			ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = 'round'; ctx.lineJoin = 'round'
			ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])))); ctx.stroke()
			ctx.globalAlpha = 1
			return pts
		}
		drawArm(prev, c.muted, 6, 0.35)
		const pts = drawArm(q, c.ink, 8, 1)
		for (const p of pts.slice(0, 3)) { ctx.fillStyle = c.bg; ctx.strokeStyle = c.ink; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 6, 0, 7); ctx.fill(); ctx.stroke() }
		// tool trail and target
		ctx.fillStyle = c.accent
		trail.forEach((p, i) => { ctx.globalAlpha = (i + 1) / trail.length * 0.6; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 2.2, 0, 7); ctx.fill() })
		ctx.globalAlpha = 1
		ctx.beginPath(); ctx.arc(X(pts[3][0]), Y(pts[3][1]), 5, 0, 7); ctx.fill()
		const gx = X(target[0]), gy = Y(target[1])
		ctx.strokeStyle = c.accent; ctx.lineWidth = 2
		ctx.beginPath(); ctx.arc(gx, gy, 10, 0, 7); ctx.moveTo(gx - 16, gy); ctx.lineTo(gx + 16, gy); ctx.moveTo(gx, gy - 16); ctx.lineTo(gx, gy + 16); ctx.stroke()
		ctx.fillStyle = c.muted
		ctx.fillText(L('dotted circle: full reach', 'noktalı çember: tam erişim'), 14, 20)
		ctx.fillText(L('drag the target', 'hedefi sürükle'), 14, 40)

		const e = sub(target, pts[3])
		const J = jac(pts)
		let a = 0, b = 0, d = 0
		for (const col of J) { a += col[0] * col[0]; b += col[0] * col[1]; d += col[1] * col[1] }
		const smin = Math.sqrt(Math.max(0, (a + d) / 2 - Math.sqrt(Math.max(0, ((a - d) / 2) ** 2 + b * b))))
		rErr(`${num(Math.hypot(e[0], e[1]) * 1000, 1)} mm`)
		rIt(String(iters))
		rDq(`${num(lastDq, 3)} rad`)
		rSig(`${num(smin, 4)} m/rad`)
		rLim(String(q.filter((qi, i) => Math.abs(Math.abs(qi) - LIM[i]) < 1e-9).length))
	})
}
