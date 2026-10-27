// Lesson 16 · static friction-cone check of a two-jaw antipodal grasp on a 3 cm cube.
// A stalled jaw servo presses with τ/r_j and the fixed jaw pushes back with the same force, so
// each contact carries N = τ/r_j and the squeeze is N1 + N2 = 2τ/r_j (the lesson's corrected
// statics). The toy jaw lever r_j = 7.8 cm is chosen so that 0.5 N·m gives about the 12.7 N
// squeeze measured in part A. Each contact must carry half the weight inside its cone.
import { animate, colors, frame, mono, type Lang } from './_kit'

const RJ = 0.078, G = 9.81

export default function (root: HTMLElement, lang: Lang) {
	const T_ = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => (lang === 'tr' ? x.toFixed(d).replace('.', ',') : x.toFixed(d))
	const f = frame(root, lang, {
		title: { en: 'Does friction carry the cube?', tr: 'Sürtünme küpü taşıyor mu?' },
		caption: {
			en: 'Two jaws squeeze a cube from the sides; nothing holds it from below. Each contact force must stay inside its friction cone (half-angle arctan μ). The arrows are what each contact has to supply: the squeeze sideways and half the weight upward. If an arrow leaves its cone, the cube slides out. Change μ, the jaw torque cap and the mass.',
			tr: 'İki çene küpü yanlardan sıkıyor; onu alttan tutan hiçbir şey yok. Her contact kuvveti kendi friction cone’unun (yarı açı arctan μ) içinde kalmalı. Oklar her contact’ın sağlaması gereken kuvvet: yana doğru sıkma ve yukarı doğru ağırlığın yarısı. Ok cone’undan çıkarsa küp kayıp düşer. μ’yü, çene torque cap’ini ve kütleyi değiştir.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 320)
	const logMu = f.slider({ en: 'friction μ', tr: 'sürtünme μ' }, { min: -2, max: 0, step: 0.01, value: Math.log10(0.3), fmt: (x) => num(10 ** x, 3) })
	const cap = f.slider({ en: 'jaw torque cap τ [N·m]', tr: 'çene torque cap’i τ [N·m]' }, { min: 0.1, max: 2.94, step: 0.01, value: 0.5, fmt: (x) => num(x, 2) })
	const mass = f.slider({ en: 'cube mass [kg]', tr: 'küp kütlesi [kg]' }, { min: 0.01, max: 0.7, step: 0.01, value: 0.2, fmt: (x) => num(x, 2) })
	const rSq = f.readout({ en: 'squeeze N1+N2', tr: 'sıkma N1+N2' })
	const rFr = f.readout({ en: 'friction can carry μ(N1+N2)', tr: 'sürtünmenin taşıyabileceği μ(N1+N2)' })
	const rW = f.readout({ en: 'weight mg', tr: 'ağırlık mg' })
	const rRatio = f.readout({ en: 'ratio', tr: 'oran' })
	const verdict = f.note({ en: '', tr: '' })

	let drop = 0, vDrop = 0, wait = 0

	animate(root, (dt) => {
		const c = colors()
		const mu = 10 ** logMu.get(), tau = cap.get(), m = mass.get()
		const N = tau / RJ, squeeze = 2 * N, ratio = (mu * squeeze) / (m * G)
		const holds = ratio >= 1
		// slipping cube: falls with the acceleration friction cannot cancel, then resets
		if (holds) { drop = Math.max(0, drop - dt * 300); vDrop = 0; wait = 0 }
		else if (drop < 170) { vDrop += dt * 900 * (1 - ratio) + dt * 40; drop += vDrop * dt }
		else if ((wait += dt) > 0.8) { drop = 0; vDrop = 0; wait = 0 }

		ctx.clearRect(0, 0, W, H)
		ctx.font = mono
		const CX = 200, CY = 140, S = 120 // cube centre and side (3 cm = 120 px)
		const yC = CY + drop
		// jaws
		ctx.fillStyle = c.ink
		ctx.fillRect(CX - S / 2 - 18, CY - S / 2 - 30, 18, S + 40)
		ctx.fillRect(CX + S / 2, CY - S / 2 - 30, 18, S + 40)
		ctx.fillStyle = c.muted
		ctx.fillText(T_('fixed jaw', 'sabit çene'), CX - S / 2 - 100, CY - S / 2 - 38)
		ctx.fillText(T_('moving jaw', 'hareketli çene'), CX + S / 2 - 10, CY - S / 2 - 38)
		// cube
		ctx.fillStyle = c.panel; ctx.strokeStyle = c.ink; ctx.lineWidth = 2
		ctx.fillRect(CX - S / 2, yC - S / 2, S, S); ctx.strokeRect(CX - S / 2, yC - S / 2, S, S)
		// friction cones at the two contacts (apex on the jaw face, opening into the cube)
		const half = Math.atan(mu), L = 95
		const need = Math.atan2(m * G / 2, N) // angle of the required force from the normal
		for (const side of [-1, 1]) {
			const ax = CX + side * S / 2, ay = CY
			const dir = -side // normal points into the cube
			ctx.fillStyle = c.soft; ctx.strokeStyle = c.accent; ctx.lineWidth = 1.2
			ctx.beginPath(); ctx.moveTo(ax, ay)
			ctx.lineTo(ax + dir * L * Math.cos(half), ay - L * Math.sin(half))
			ctx.lineTo(ax + dir * L * Math.cos(half), ay + L * Math.sin(half))
			ctx.closePath(); ctx.fill(); ctx.stroke()
			// required contact force: squeeze along the normal, half the weight upward
			const len = Math.min(110, 80 / Math.cos(need))
			const ex = ax + dir * len * Math.cos(need), ey = ay - len * Math.sin(need)
			ctx.strokeStyle = holds ? c.accent : c.ink; ctx.lineWidth = 3
			ctx.setLineDash(holds ? [] : [6, 4])
			ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ex, ey); ctx.stroke(); ctx.setLineDash([])
			const ang = Math.atan2(ey - ay, ex - ax)
			ctx.fillStyle = holds ? c.accent : c.ink
			ctx.beginPath(); ctx.moveTo(ex, ey)
			ctx.lineTo(ex - 10 * Math.cos(ang - 0.4), ey - 10 * Math.sin(ang - 0.4))
			ctx.lineTo(ex - 10 * Math.cos(ang + 0.4), ey - 10 * Math.sin(ang + 0.4)); ctx.fill()
		}
		// weight
		ctx.strokeStyle = c.ink; ctx.lineWidth = 2.5
		ctx.beginPath(); ctx.moveTo(CX, yC + 10); ctx.lineTo(CX, yC + 48); ctx.stroke()
		ctx.fillStyle = c.ink
		ctx.beginPath(); ctx.moveTo(CX - 6, yC + 42); ctx.lineTo(CX + 6, yC + 42); ctx.lineTo(CX, yC + 54); ctx.fill()
		ctx.fillText('mg', CX + 8, yC + 40)
		ctx.fillStyle = c.muted
		ctx.fillText(`${T_('cone', 'cone')} ±${num((half * 180) / Math.PI, 1)}°`, CX - S / 2 - 100, H - 14)

		// ratio on a log axis, with the measured boundary band of part C
		const bx0 = 430, bx1 = W - 20, by = 150
		const lx = (r: number) => bx0 + ((Math.log10(Math.min(100, Math.max(0.1, r))) + 1) / 3) * (bx1 - bx0)
		ctx.fillStyle = c.soft; ctx.fillRect(lx(1.12), by - 26, lx(1.24) - lx(1.12), 52)
		ctx.strokeStyle = c.line; ctx.lineWidth = 1.5
		ctx.beginPath(); ctx.moveTo(bx0, by); ctx.lineTo(bx1, by); ctx.stroke()
		ctx.fillStyle = c.muted
		for (const r of [0.1, 1, 10, 100]) {
			ctx.beginPath(); ctx.moveTo(lx(r), by - 5); ctx.lineTo(lx(r), by + 5); ctx.stroke()
			const t = num(r, r < 1 ? 1 : 0)
			ctx.fillText(t, lx(r) - ctx.measureText(t).width / 2, by + 22)
		}
		ctx.strokeStyle = c.ink; ctx.setLineDash([4, 4])
		ctx.beginPath(); ctx.moveTo(lx(1), by - 40); ctx.lineTo(lx(1), by + 30); ctx.stroke(); ctx.setLineDash([])
		ctx.fillText('μ(N1+N2) / mg', bx0, by - 52)
		ctx.fillStyle = holds ? c.accent : c.ink
		ctx.beginPath(); ctx.arc(lx(ratio), by, 8, 0, 7); ctx.fill()
		ctx.fillStyle = c.muted
		ctx.fillText(T_('slips', 'kayar'), bx0, by + 50)
		const hl = T_('holds', 'tutar')
		ctx.fillText(hl, bx1 - ctx.measureText(hl).width, by + 50)
		ctx.fillText(T_('shaded: part C boundary', 'gölgeli: C bölümü sınırı'), bx0, by + 76)

		rSq(num(squeeze, 2) + ' N')
		rFr(num(mu * squeeze, 2) + ' N')
		rW(num(m * G, 2) + ' N')
		rRatio(num(ratio, 2))
		verdict(
			mu <= 0.03
				? T_('Static test only. In part A, at μ ≤ 0.03 most grasps never formed: the cube left the jaws while they were closing.',
					'Yalnızca statik test. A bölümünde μ ≤ 0,03’te grasp’lerin çoğu hiç oluşmadı: küp çeneler kapanırken onları terk etti.')
				: !holds
					? T_('The required force leaves the cone: friction cannot carry the weight and the cube slides down.',
						'Gereken kuvvet cone’dan çıkıyor: sürtünme ağırlığı taşıyamıyor ve küp aşağı kayıyor.')
					: ratio < 1.24
						? T_('Inside the cone, but barely. Lifting also accelerates the cube; in part C ratios of 1.12 and below dropped during the lift.',
							'Cone’un içinde ama sınırda. Kaldırma küpü ayrıca ivmelendirir; C bölümünde 1,12 ve altındaki oranlar kaldırma sırasında düştü.')
						: T_('Inside the cone with margin: the cube holds.', 'Cone’un içinde, payı var: küp tutuluyor.'),
		)
	})
}
