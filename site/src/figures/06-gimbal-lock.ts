// Lesson 06 · a three-ring gimbal for intrinsic z-y'-x'' Euler angles, R = Rz(yaw) Ry(pitch) Rx(roll).
// The outer ring turns about the fixed vertical (yaw) axis, the middle ring about the pitch axis it
// carries, the inner ring about the roll axis. At pitch ±90° the roll axis lines up with the yaw axis
// and one direction of rotation is lost. Readouts use the lesson's Euler-rate map E (det E = cos θ).
import { animate, colors, frame, mono, type Lang } from './_kit'

type V = [number, number, number]
type M = [V, V, V] // rows

const mul = (A: M, B: M): M => A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j])) as M
const app = (A: M, v: V): V => [A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2], A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2], A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2]]
const Rz = (a: number): M => [[Math.cos(a), -Math.sin(a), 0], [Math.sin(a), Math.cos(a), 0], [0, 0, 1]]
const Ry = (a: number): M => [[Math.cos(a), 0, Math.sin(a)], [0, 1, 0], [-Math.sin(a), 0, Math.cos(a)]]
const Rx = (a: number): M => [[1, 0, 0], [0, Math.cos(a), -Math.sin(a)], [0, Math.sin(a), Math.cos(a)]]

/** eigenvalues of a symmetric 3×3 matrix (cyclic Jacobi rotations) */
function symEig(S: M): number[] {
	const A = S.map((r) => [...r])
	for (let sweep = 0; sweep < 30; sweep++) {
		for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
			if (Math.abs(A[p][q]) < 1e-300) continue
			const th = (A[q][q] - A[p][p]) / (2 * A[p][q])
			const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1))
			const c = 1 / Math.sqrt(t * t + 1), s = t * c
			for (let k = 0; k < 3; k++) { const a = A[k][p], b = A[k][q]; A[k][p] = c * a - s * b; A[k][q] = s * a + c * b }
			for (let k = 0; k < 3; k++) { const a = A[p][k], b = A[q][k]; A[p][k] = c * a - s * b; A[q][k] = s * a + c * b }
		}
	}
	return [A[0][0], A[1][1], A[2][2]]
}

export default function (root: HTMLElement, lang: Lang) {
	const L = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => x.toFixed(d).replace('.', L('.', ','))
	const f = frame(root, lang, {
		title: { en: 'Where the third knob goes missing', tr: 'Üçüncü düğmenin kaybolduğu yer' },
		caption: {
			en: 'Three rings, three Euler angles: yaw turns the outer ring about the vertical, pitch the middle ring, roll the inner ring and the body. Raise pitch towards 90° and watch the roll axis swing onto the yaw axis. Then wiggle yaw and wiggle roll: near 90° they make the same motion, and no knob turns the body sideways.',
			tr: 'Üç halka, üç Euler açısı: yaw dış halkayı dikey eksen etrafında, pitch ortadaki halkayı, roll iç halkayı ve gövdeyi döndürür. Pitch’i 90°’ye doğru yükselt ve roll ekseninin yaw ekseninin üstüne kaydığını izle. Sonra yaw’ı ve roll’u salla: 90°’ye yakın ikisi aynı hareketi yapar ve gövdeyi yana çeviren düğme kalmaz.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 340)
	const deg = (x: number) => `${num(x, 0)}°`
	const yaw = f.slider({ en: 'yaw ψ', tr: 'yaw ψ' }, { min: -180, max: 180, step: 1, value: 30, fmt: deg })
	const pitch = f.slider({ en: 'pitch θ', tr: 'pitch θ' }, { min: -90, max: 90, step: 0.5, value: 60, fmt: (x) => `${num(x, 1)}°` })
	const roll = f.slider({ en: 'roll φ', tr: 'roll φ' }, { min: -180, max: 180, step: 1, value: 30, fmt: deg })
	const wig = f.select({ en: 'wiggle', tr: 'salla' }, [{ en: 'nothing', tr: 'hiçbiri' }, { en: 'yaw ±25°', tr: 'yaw ±25°' }, { en: 'roll ±25°', tr: 'roll ±25°' }])
	f.button({ en: 'Pitch to 90°', tr: 'Pitch’i 90° yap' }, () => pitch.set(90))
	const rDet = f.readout({ en: 'det E = cos θ', tr: 'det E = cos θ' })
	const rCond = f.readout({ en: 'cond(E)', tr: 'cond(E)' })
	const rAng = f.readout({ en: 'angle yaw axis ↔ roll axis', tr: 'yaw ekseni ↔ roll ekseni açısı' })
	const rRate = f.readout({ en: 'Euler rate for 1 rad/s sideways', tr: '1 rad/s yana dönüş için Euler hızı' })

	// orthographic camera: turn the scene about z, then tilt it towards the viewer
	const az = (35 * Math.PI) / 180, el = (18 * Math.PI) / 180, S = 118, cx = 250, cy = 175
	const proj = (p: V) => {
		const x1 = p[0] * Math.cos(az) - p[1] * Math.sin(az), y1 = p[0] * Math.sin(az) + p[1] * Math.cos(az)
		const depth = x1 * Math.cos(el) + p[2] * Math.sin(el), up = -x1 * Math.sin(el) + p[2] * Math.cos(el)
		return { x: cx + S * y1, y: cy - S * up, d: depth }
	}
	let t = 0

	animate(root, (dt) => {
		t += dt
		const w = 25 * Math.sin(t * 2.2)
		const psi = ((yaw.get() + (wig.get() === 1 ? w : 0)) * Math.PI) / 180
		const th = (pitch.get() * Math.PI) / 180
		const phi = ((roll.get() + (wig.get() === 2 ? w : 0)) * Math.PI) / 180
		const A1 = Rz(psi), A2 = mul(A1, Ry(th)), A3 = mul(A2, Rx(phi))
		const c = colors()
		ctx.clearRect(0, 0, W, H)
		ctx.font = mono

		const ring = (Rm: M, r: number, plane: 'yz' | 'xy' | 'xz', color: string, width: number) => {
			const N = 72
			const pts = Array.from({ length: N + 1 }, (_, i) => {
				const a = (i / N) * 2 * Math.PI, u = r * Math.cos(a), v = r * Math.sin(a)
				const local: V = plane === 'yz' ? [0, u, v] : plane === 'xy' ? [u, v, 0] : [u, 0, v]
				return proj(app(Rm, local))
			})
			ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = 'round'
			for (let i = 0; i < N; i++) {
				ctx.globalAlpha = (pts[i].d + pts[i + 1].d) / 2 >= 0 ? 1 : 0.3
				ctx.beginPath(); ctx.moveTo(pts[i].x, pts[i].y); ctx.lineTo(pts[i + 1].x, pts[i + 1].y); ctx.stroke()
			}
			ctx.globalAlpha = 1
		}
		const axis = (dir: V, len: number, color: string, label: string) => {
			const a = proj([dir[0] * len, dir[1] * len, dir[2] * len]), b = proj([-dir[0] * len, -dir[1] * len, -dir[2] * len])
			ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.setLineDash([6, 4])
			ctx.beginPath(); ctx.moveTo(b.x, b.y); ctx.lineTo(a.x, a.y); ctx.stroke(); ctx.setLineDash([])
			ctx.fillStyle = color; ctx.beginPath(); ctx.arc(a.x, a.y, 3.5, 0, 7); ctx.fill()
			ctx.fillText(label, a.x + 6, a.y - 6)
		}
		// fixed base: a stand under the outer ring
		const foot = proj([0, 0, -1.25]), base = proj([0, 0, -1.05])
		ctx.strokeStyle = c.line; ctx.lineWidth = 6; ctx.beginPath(); ctx.moveTo(foot.x, foot.y + 12); ctx.lineTo(base.x, base.y); ctx.stroke()
		ring(A1, 1.05, 'yz', c.muted, 5) // yaw ring: pivots on the vertical axis, carries the pitch bearings
		ring(A2, 0.85, 'xy', c.ink, 4) // pitch ring: carries the roll bearings
		ring(A3, 0.66, 'xz', c.accent, 4) // roll ring, fixed to the body
		// the body: a nose arrow along body x and a fin along body z
		const nose = proj(app(A3, [0.5, 0, 0])), tail = proj(app(A3, [-0.35, 0, 0])), fin = proj(app(A3, [-0.25, 0, 0.25])), wl = proj(app(A3, [-0.2, 0.22, 0])), wr = proj(app(A3, [-0.2, -0.22, 0]))
		ctx.strokeStyle = c.ink; ctx.lineWidth = 5
		ctx.beginPath(); ctx.moveTo(tail.x, tail.y); ctx.lineTo(nose.x, nose.y); ctx.stroke()
		ctx.lineWidth = 3
		ctx.beginPath(); ctx.moveTo(wl.x, wl.y); ctx.lineTo(wr.x, wr.y); ctx.moveTo(tail.x, tail.y); ctx.lineTo(fin.x, fin.y); ctx.stroke()
		ctx.fillStyle = c.accent; ctx.beginPath(); ctx.arc(nose.x, nose.y, 6, 0, 7); ctx.fill()
		// rotation axes
		const zAxis: V = [0, 0, 1], yAxis = app(A1, [0, 1, 0]), xAxis = app(A2, [1, 0, 0])
		axis(zAxis, 1.32, c.muted, L('yaw axis', 'yaw ekseni'))
		axis(yAxis, 1.2, c.ink, L('pitch axis', 'pitch ekseni'))
		axis(xAxis, 1.15, c.accent, L('roll axis', 'roll ekseni'))

		// readouts from E(θ, φ): ω_body = E [φ', θ', ψ']
		const s = Math.sin(th), co = Math.cos(th), sp = Math.sin(phi), cp = Math.cos(phi)
		const E: M = [[1, 0, -s], [0, cp, sp * co], [0, -sp, cp * co]]
		const Et: M = [[E[0][0], E[1][0], E[2][0]], [E[0][1], E[1][1], E[2][1]], [E[0][2], E[1][2], E[2][2]]]
		const ev = symEig(mul(Et, E)).map((x) => Math.max(0, x))
		const lo = Math.min(...ev), hi = Math.max(...ev)
		const locked = Math.abs(co) < 1e-9
		rDet(num(co, 4))
		rCond(locked || lo <= 1e-24 ? '∞' : num(Math.sqrt(hi / lo), 2))
		const ang = (Math.acos(Math.min(1, Math.abs(xAxis[2]))) * 180) / Math.PI
		rAng(`${num(ang, 1)}°`)
		// lost direction at θ = 90°: (0, sin φ, cos φ); solving E α = ω gives |α| = sqrt(1 + sin²θ) / |cos θ|
		rRate(locked ? '∞' : `${num(Math.sqrt(1 + s * s) / Math.abs(co), 1)} rad/s`)
		ctx.fillStyle = c.muted
		ctx.fillText(locked ? L('locked: roll axis = yaw axis', 'kilitli: roll ekseni = yaw ekseni') : ang < 10 ? L('almost locked', 'neredeyse kilitli') : '', 430, 24)
	})
}
