// Lesson 19 · why a flat board cannot pin down the focal length. The lesson's camera (640×480,
// fovy 50° → f = 514.68 px, cx = 319.5, cy = 239.5) sees a 9×6 checkerboard with 25 mm squares
// 0.575 m away, tilted about its vertical axis. For a wrong focal length f, we re-fit the board
// pose (all six degrees of freedom, Levenberg–Marquardt on the corner pixels, as solvePnP does)
// and report how well that wrong camera still explains the true corners. No pixel noise.
import { animate, axes, colors, frame, mono, type Lang } from './_kit'

const F0 = 514.68, CX = 319.5, CY = 239.5, Z0 = 0.575
const CORNERS: [number, number][] = []
for (let j = 0; j < 6; j++) for (let i = 0; i < 9; i++) CORNERS.push([(i - 4) * 0.025, (j - 2.5) * 0.025])

type Pose = number[] // [rx, ry, rz, tx, ty, tz], rotation vector + translation (camera frame)

function rot(rx: number, ry: number, rz: number) {
	const th = Math.hypot(rx, ry, rz)
	if (th < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1]
	const x = rx / th, y = ry / th, z = rz / th, c = Math.cos(th), s = Math.sin(th), C = 1 - c
	return [c + x * x * C, x * y * C - z * s, x * z * C + y * s, y * x * C + z * s, c + y * y * C, y * z * C - x * s, z * x * C - y * s, z * y * C + x * s, c + z * z * C]
}

function project(p: Pose, f: number, out: Float64Array) {
	const R = rot(p[0], p[1], p[2])
	CORNERS.forEach(([X, Y], k) => {
		const x = R[0] * X + R[1] * Y + p[3], y = R[3] * X + R[4] * Y + p[4], z = R[6] * X + R[7] * Y + p[5]
		out[2 * k] = (f * x) / z + CX
		out[2 * k + 1] = (f * y) / z + CY
	})
}

function solve(A: number[][], b: number[]) {
	// Gaussian elimination with partial pivoting for the 6×6 normal equations
	const n = b.length, M = A.map((r, i) => [...r, b[i]])
	for (let c = 0; c < n; c++) {
		let piv = c
		for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r
		;[M[c], M[piv]] = [M[piv], M[c]]
		const d = M[c][c] || 1e-18
		for (let r = c + 1; r < n; r++) { const k = M[r][c] / d; for (let j = c; j <= n; j++) M[r][j] -= k * M[c][j] }
	}
	const x = new Array(n).fill(0)
	for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let j = r + 1; j < n; j++) s -= M[r][j] * x[j]; x[r] = s / (M[r][r] || 1e-18) }
	return x
}

/** Best board pose for focal length f against the reference pixels; returns pose and RMS px. */
function fit(f: number, ref: Float64Array, init: Pose) {
	const m = ref.length, cur = new Float64Array(m), tmp = new Float64Array(m)
	let p = [...init], lambda = 1e-3
	const cost = (q: Pose) => { project(q, f, tmp); let s = 0; for (let i = 0; i < m; i++) s += (tmp[i] - ref[i]) ** 2; return s }
	let e = cost(p)
	for (let it = 0; it < 40; it++) {
		project(p, f, cur)
		const J: number[][] = []
		for (let k = 0; k < 6; k++) {
			const q = [...p]; const h = k < 3 ? 1e-6 : 1e-7
			q[k] += h; project(q, f, tmp)
			J.push(Array.from(tmp, (v, i) => (v - cur[i]) / h))
		}
		const A = J.map((a) => J.map((b) => a.reduce((s, v, i) => s + v * b[i], 0)))
		const g = J.map((a) => a.reduce((s, v, i) => s + v * (cur[i] - ref[i]), 0))
		let improved = false
		for (let tries = 0; tries < 8 && !improved; tries++) {
			const Ad = A.map((r, i) => r.map((v, j) => (i === j ? v * (1 + lambda) + 1e-12 : v)))
			const d = solve(Ad, g.map((v) => -v))
			const q = p.map((v, i) => v + d[i])
			const eq = q[5] > 0.01 ? cost(q) : Infinity
			if (eq < e) { improved = e - eq > 1e-10 * e; p = q; e = eq; lambda = Math.max(1e-9, lambda / 5); if (!improved) break } else lambda *= 10
		}
		if (!improved || e < 1e-20) break
	}
	return { pose: p, rms: Math.sqrt(e / (m / 2)) }
}

export default function (root: HTMLElement, lang: Lang) {
	const T_ = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => (lang === 'tr' ? x.toFixed(d).replace('.', ',') : x.toFixed(d))
	const f = frame(root, lang, {
		title: { en: 'Long lens far away, or short lens close?', tr: 'Uzak ve uzun lens mi, yakın ve kısa lens mi?' },
		caption: {
			en: 'One board view; the true camera has f = 514.68 px. Guess a different focal length: the board pose is re-fitted to the true corner pixels, and the right panel shows how well that wrong camera still explains them. With the board facing the camera (tilt 0°) every focal length fits perfectly; tilt the board and wrong guesses start to leave a measurable error.',
			tr: 'Tek bir board görüntüsü; gerçek kameranın f’si 514,68 px. Farklı bir focal length tahmin et: board pozu gerçek köşe pixel’lerine yeniden oturtuluyor; sağ panel bu yanlış kameranın onları hâlâ ne kadar iyi açıkladığını gösteriyor. Board kameraya dümdüz bakarken (tilt 0°) her focal length kusursuz oturuyor; board’u eğ, yanlış tahminler ölçülebilir bir hata bırakmaya başlasın.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 320)
	const tilt = f.slider({ en: 'board tilt [°]', tr: 'board tilt [°]' }, { min: 0, max: 35, step: 1, value: 0 })
	const focal = f.slider({ en: 'focal length guess f [px]', tr: 'focal length tahmini f [px]' }, { min: 260, max: 1500, step: 5, value: 800 })
	const rZ = f.readout({ en: 'fitted board distance', tr: 'oturtulan board mesafesi' })
	const rT = f.readout({ en: 'fitted tilt', tr: 'oturtulan tilt' })
	const rRms = f.readout({ en: 'reprojection RMS', tr: 'reprojection RMS' })
	const note = f.note({ en: '', tr: '' })

	const ref = new Float64Array(CORNERS.length * 2)
	const FGRID = Array.from({ length: 41 }, (_, i) => 260 * (1500 / 260) ** (i / 40))
	let curve: [number, number][] = []
	let cur = { pose: [0, 0, 0, 0, 0, Z0] as Pose, rms: 0 }
	const truePose = (): Pose => [0, (tilt.get() * Math.PI) / 180, 0, 0, 0, Z0]
	const fitAt = (fv: number) => {
		const t = truePose()
		return fit(fv, ref, [t[0], t[1], t[2], 0, 0, (Z0 * fv) / F0])
	}
	const refit = () => { cur = fitAt(focal.get()) }
	const retilt = () => {
		project(truePose(), F0, ref)
		curve = FGRID.map((fv) => [Math.log10(fv), Math.log10(Math.max(1e-3, fitAt(fv).rms))])
		refit()
	}
	retilt()
	tilt.on(retilt); focal.on(refit)

	animate(root, () => {
		const c = colors()
		ctx.clearRect(0, 0, W, H)
		ctx.font = mono
		// side view from above: camera at the left, optical axis to the right; 1 m = 190 px
		const ox = 30, oy = 150, S = 190
		ctx.strokeStyle = c.line; ctx.lineWidth = 1; ctx.setLineDash([3, 4])
		ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + 1.75 * S, oy); ctx.stroke(); ctx.setLineDash([])
		const fv = focal.get()
		// image plane at a distance proportional to f
		const ipx = ox + (fv / 1500) * 40
		ctx.strokeStyle = c.ink; ctx.lineWidth = 2
		ctx.beginPath(); ctx.moveTo(ox, oy - 14); ctx.lineTo(ox, oy + 14); ctx.lineTo(ox - 14, oy); ctx.closePath(); ctx.stroke()
		ctx.strokeStyle = c.accent; ctx.beginPath(); ctx.moveTo(ipx, oy - 20); ctx.lineTo(ipx, oy + 20); ctx.stroke()
		ctx.fillStyle = c.muted
		ctx.fillText(`f = ${fv} px`, ox - 10, oy + 40)
		// a board seen from above is a segment of half-width 0.1 m
		const board = (p: Pose, color: string, dash: number[], width: number) => {
			const R = rot(p[0], p[1], p[2])
			const pt = (X: number) => [ox + (R[6] * X + p[5]) * S, oy + (R[0] * X + p[3]) * S]
			const [a0, a1] = pt(-0.1), [b0, b1] = pt(0.1)
			ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash)
			ctx.beginPath(); ctx.moveTo(a0, a1); ctx.lineTo(b0, b1); ctx.stroke(); ctx.setLineDash([])
			ctx.strokeStyle = c.line; ctx.lineWidth = 1
			for (const [x, y] of [[a0, a1], [b0, b1]]) { ctx.beginPath(); ctx.moveTo(ipx, oy); ctx.lineTo(x, y); ctx.stroke() }
		}
		board(truePose(), c.ink, [5, 4], 2.5)
		board(cur.pose, c.accent, [], 4)
		ctx.fillStyle = c.muted
		ctx.fillText(T_('dashed: true board', 'kesikli: gerçek board'), ox, 22)
		ctx.fillText(T_('orange: fit for this f', 'turuncu: bu f için oturtulan'), ox, 40)
		ctx.fillText(T_('top view', 'üstten görünüş'), ox, H - 14)

		// right: best-fit RMS against the focal-length guess, log-log
		const b = axes(ctx, { x0: 380, y0: 20, x1: W - 14, y1: H - 34 }, [Math.log10(260), Math.log10(1500)], [-3, 2])
		b.draw(c, T_('f guess [px] →', 'f tahmini [px] →'), T_('RMS [px]', 'RMS [px]'))
		ctx.fillStyle = c.muted; ctx.strokeStyle = c.line; ctx.lineWidth = 1
		for (const d of [-3, -2, -1, 0, 1, 2]) {
			const s = num(10 ** d, Math.max(0, -d))
			ctx.fillText(s, 376 - ctx.measureText(s).width, b.Y(d) + 4)
		}
		for (const t of [300, 514.68, 1000, 1500]) {
			const X = b.X(Math.log10(t)), s = t === 514.68 ? 'f₀' : String(t)
			ctx.beginPath(); ctx.moveTo(X, H - 34); ctx.lineTo(X, H - 29); ctx.stroke()
			ctx.fillText(s, X - ctx.measureText(s).width / 2, H - 16)
		}
		b.line([[Math.log10(260), Math.log10(0.09)], [Math.log10(1500), Math.log10(0.09)]], c.muted, 1.2, [5, 5])
		ctx.fillText('0.09 px'.replace('.', lang === 'tr' ? ',' : '.'), W - 80, b.Y(Math.log10(0.09)) - 6)
		b.line(curve, c.accent, 2.5)
		ctx.fillStyle = c.accent
		ctx.beginPath(); ctx.arc(b.X(Math.log10(fv)), b.Y(Math.log10(Math.max(1e-3, cur.rms))), 6, 0, 7); ctx.fill()

		rZ(num(cur.pose[5] * 100, 1) + ' cm')
		const R = rot(cur.pose[0], cur.pose[1], cur.pose[2])
		rT(num((Math.acos(Math.min(1, Math.abs(R[8]))) * 180) / Math.PI, 1) + '°')
		rRms(cur.rms < 1e-3 ? '< ' + num(0.001, 3) + ' px' : num(cur.rms, 3) + ' px')
		note(cur.rms < 0.09
			? T_('This wrong camera explains the corners within about the lesson’s training RMS (0.09 px): the error number cannot tell it apart.',
				'Bu yanlış kamera köşeleri dersin eğitim RMS’i (0,09 px) civarında açıklıyor: hata sayısı onu ayırt edemez.')
			: T_('The corners give this focal length away: no board pose explains them within 0.09 px.',
				'Köşeler bu focal length’i ele veriyor: hiçbir board pozu onları 0,09 px içinde açıklamıyor.'))
	})
}
