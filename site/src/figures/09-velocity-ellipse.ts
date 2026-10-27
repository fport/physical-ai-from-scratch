// Lesson 09 · the SO-101's three pitch joints in the arm plane (geometry and ranges from the Menagerie
// model, pan = 0, wrist_roll = 0). The 2×3 position Jacobian maps joint speeds to tool velocity; the
// ellipse is every tool velocity reachable with |q̇| = 1 rad/s. Its transpose maps a tool force to
// joint torques: for a downward force each torque equals that joint's horizontal lever to the tool.
import { animate, colors, frame, mono, type Lang } from './_kit'

const O1 = [0.06923, 0.1166], O2 = [0.09723, 0.22917], O3 = [0.23213, 0.23437], TIP = [0.39136, 0.24635]
const D = Math.PI / 180

const rot = (v: number[], q: number) => [v[0] * Math.cos(q) + v[1] * Math.sin(q), -v[0] * Math.sin(q) + v[1] * Math.cos(q)]
const add = (a: number[], b: number[]) => [a[0] + b[0], a[1] + b[1]]
const sub = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1]]

function fk(q: number[]) {
	const p2 = add(O1, rot(sub(O2, O1), q[0]))
	const p3 = add(p2, rot(sub(O3, O2), q[0] + q[1]))
	const pt = add(p3, rot(sub(TIP, O3), q[0] + q[1] + q[2]))
	return [O1, p2, p3, pt]
}

/** column i = (axis +y) × (tool − joint i), written in (x, z): (r_z, −r_x) */
function jacobian(pts: number[][]) {
	const t = pts[3]
	return [0, 1, 2].map((i) => [t[1] - pts[i][1], -(t[0] - pts[i][0])])
}

/** singular values and left singular vectors of a 2×n matrix given by its columns */
function svd2(cols: number[][]) {
	let a = 0, b = 0, d = 0 // J Jᵀ = [[a, b], [b, d]]
	for (const c of cols) { a += c[0] * c[0]; b += c[0] * c[1]; d += c[1] * c[1] }
	const tr = a + d, det = a * d - b * b, disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det))
	const l1 = tr / 2 + disc, l2 = Math.max(0, tr / 2 - disc)
	const ang = 0.5 * Math.atan2(2 * b, a - d) // direction of the largest stretch
	return { s1: Math.sqrt(l1), s2: Math.sqrt(l2), ang }
}

export default function (root: HTMLElement, lang: Lang) {
	const L = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => x.toFixed(d).replace('.', L('.', ','))
	const f = frame(root, lang, {
		title: { en: 'The velocity ellipse', tr: 'Hız elipsi' },
		caption: {
			en: 'Turn the three pitch joints at a combined 1 rad/s in every possible mix: the tool’s velocity fills this ellipse. Stretch the arm and the ellipse flattens; straight out it cannot move outward at all. The same Jacobian, transposed, turns a downward push on the tool into joint torques, and the lever grows as the arm reaches out.',
			tr: 'Üç pitch joint’ini toplamda 1 rad/s ile mümkün olan her karışımda döndür: aletin hızı bu elipsi doldurur. Kolu ger, elips yassılaşır; dümdüz uzanınca dışarı doğru hiç hareket edemez. Aynı Jacobian’ın transpose’u, alete aşağı doğru bir itmeyi joint torklarına çevirir ve kol uzandıkça kaldıraç büyür.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 340)
	const deg = (x: number) => `${num(x, 1)}°`
	const q1 = f.slider({ en: 'shoulder_lift', tr: 'shoulder_lift' }, { min: -100, max: 100, step: 0.1, value: -34.4, fmt: deg })
	const q2 = f.slider({ en: 'elbow_flex', tr: 'elbow_flex' }, { min: -96.8, max: 96.8, step: 0.1, value: 51.6, fmt: deg })
	const q3 = f.slider({ en: 'wrist_flex', tr: 'wrist_flex' }, { min: -95, max: 95, step: 0.1, value: 22.9, fmt: deg })
	const cols = f.checkbox({ en: 'show the 3 columns', tr: '3 sütunu göster' }, true)
	// the lesson's mid (0, −0.6, 0.9, 0.4) and stretched (0, 1.3, −1.3, 0) poses, then fully straight
	f.button({ en: 'Mid pose', tr: 'Orta poz' }, () => { q1.set(-34.4); q2.set(51.6); q3.set(22.9) })
	f.button({ en: 'Stretched pose', tr: 'Gergin poz' }, () => { q1.set(74.5); q2.set(-74.5); q3.set(0) })
	f.button({ en: 'Straight', tr: 'Dümdüz' }, () => { q1.set(74.5); q2.set(-73.8); q3.set(2.1) })
	const rS = f.readout({ en: 'σ₁, σ₂ [m/rad]', tr: 'σ₁, σ₂ [m/rad]' })
	const rW = f.readout({ en: 'w = σ₁σ₂', tr: 'w = σ₁σ₂' })
	const rCond = f.readout({ en: 'cond', tr: 'cond' })
	const rTau = f.readout({ en: 'torque per 1 N down (lift, elbow, wrist)', tr: '1 N aşağı itme başına tork (lift, elbow, wrist)' })

	const S = 420, ox = 150, oy = 250
	const X = (x: number) => ox + x * S, Y = (z: number) => oy - z * S
	const VS = 200 // px per (m/s) for the ellipse and the columns
	let t = 0

	animate(root, (dt) => {
		t += dt
		const q = [q1.get() * D, q2.get() * D, q3.get() * D]
		const pts = fk(q), J = jacobian(pts), sv = svd2(J)
		const c = colors()
		ctx.clearRect(0, 0, W, H)
		ctx.font = mono
		ctx.strokeStyle = c.line; ctx.lineWidth = 1.5
		ctx.beginPath(); ctx.moveTo(10, Y(0)); ctx.lineTo(W - 10, Y(0)); ctx.stroke()
		ctx.fillStyle = c.line; ctx.fillRect(X(0) - 6, Y(O1[1]), 50, O1[1] * S)
		// arm
		ctx.strokeStyle = c.ink; ctx.lineWidth = 8; ctx.lineCap = 'round'; ctx.lineJoin = 'round'
		ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])))); ctx.stroke()
		for (const p of pts.slice(0, 3)) { ctx.fillStyle = c.bg; ctx.strokeStyle = c.ink; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 6, 0, 7); ctx.fill(); ctx.stroke() }
		const tx = X(pts[3][0]), ty = Y(pts[3][1])
		// lever of the lift joint for a downward force: horizontal distance lift axis → tool
		ctx.strokeStyle = c.muted; ctx.lineWidth = 1.2; ctx.setLineDash([4, 4])
		ctx.beginPath(); ctx.moveTo(X(O1[0]), Y(O1[1])); ctx.lineTo(X(O1[0]), ty); ctx.lineTo(tx, ty); ctx.stroke(); ctx.setLineDash([])
		// ellipse
		const u = [Math.cos(sv.ang), Math.sin(sv.ang)], v = [-u[1], u[0]]
		ctx.fillStyle = c.soft; ctx.strokeStyle = c.accent; ctx.lineWidth = 2.5
		ctx.beginPath()
		for (let i = 0; i <= 80; i++) {
			const a = (i / 80) * 2 * Math.PI
			const ex = sv.s1 * Math.cos(a) * u[0] + sv.s2 * Math.sin(a) * v[0]
			const ez = sv.s1 * Math.cos(a) * u[1] + sv.s2 * Math.sin(a) * v[1]
			const x = tx + VS * ex, y = ty - VS * ez
			i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)
		}
		ctx.closePath(); ctx.globalAlpha = 0.7; ctx.fill(); ctx.globalAlpha = 1; ctx.stroke()
		if (cols.get()) {
			J.forEach((col, i) => {
				const x = tx + VS * col[0], y = ty - VS * col[1]
				ctx.strokeStyle = c.ink; ctx.lineWidth = 1.5
				ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(x, y); ctx.stroke()
				ctx.fillStyle = c.ink; ctx.fillText(String(i + 1), x + 4, y - 4)
			})
		}
		// a joint-speed mix sweeping the unit sphere's equator in the (u, v) directions: one tool velocity
		const a = t * 1.3
		const vx = sv.s1 * Math.cos(a) * u[0] + sv.s2 * Math.sin(a) * v[0], vz = sv.s1 * Math.cos(a) * u[1] + sv.s2 * Math.sin(a) * v[1]
		ctx.fillStyle = c.accent; ctx.beginPath(); ctx.arc(tx + VS * vx, ty - VS * vz, 4, 0, 7); ctx.fill()
		ctx.fillStyle = c.accent; ctx.beginPath(); ctx.arc(tx, ty, 5, 0, 7); ctx.fill()
		ctx.fillStyle = c.muted
		ctx.fillText(L('ellipse = tool velocity at |q̇| = 1 rad/s · 1–3 = columns of J', 'elips = |q̇| = 1 rad/s’de alet hızı · 1–3 = J’nin sütunları'), 14, 20)
		ctx.fillText(L('dashed: lift lever', 'kesikli: lift kaldıracı'), 14, 40)

		rS(`${num(sv.s1, 3)}, ${num(sv.s2, 4)}`)
		rW(sv.s1 * sv.s2 < 1e-3 ? (sv.s1 * sv.s2).toExponential(2).replace('.', L('.', ',')) : num(sv.s1 * sv.s2, 4))
		rCond(sv.s2 < 1e-9 ? '∞' : num(sv.s1 / sv.s2, 1))
		// τ = Jᵀ F with F = (0, −1) N: τ_i = −J_z,i = horizontal lever of joint i
		rTau(J.map((col) => num(-col[1], 3)).join(' · ') + ' N·m')
	})
}
