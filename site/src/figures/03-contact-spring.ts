// Lesson 03 · a 1 kg body dropped 10 cm onto a plane at dt = 2 ms, with one soft contact built
// from solref (timeconst, dampratio) and the default solimp, integrated with semi-implicit Euler.
// The refsafe checkbox applies MuJoCo's guard: timeconst used = max(timeconst, 2·dt).
import { animate, axes, colors, frame, mono, type Lang } from './_kit'

const TCS = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.05, 0.1] // s, values from the lesson
const DRS = [0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2]
const g = 9.81, h = 0.002, DROP = 0.1, T_END = 0.8

const impedance = (p: number) => { // solimp 0.9 0.95 0.001 0.5 2
	const x = Math.min(1, Math.max(0, p / 0.001))
	const y = x < 0.5 ? 0.5 * (x / 0.5) ** 2 : 1 - 0.5 * ((1 - x) / 0.5) ** 2
	return 0.9 + 0.05 * y
}

function simulate(timeconst: number, dampratio: number, refsafe: boolean) {
	const tc = refsafe ? Math.max(timeconst, 2 * h) : timeconst
	let z = DROP, v = 0, peak = 0, rebound = 0, touched = false
	const zs: number[] = [z]
	for (let t = 0; t < T_END; t += h) {
		let a = -g
		if (z < 0) { // z = height of the bottom face; below 0 it overlaps the plane by p = -z
			const p = -z, d = impedance(p), dw = 0.95
			const b = 2 / (dw * tc), k = d / (dw * dw * tc * tc * dampratio * dampratio)
			a = Math.max(-g, (1 - d) * -g + d * (-b * v + k * p)) // push only
			touched = true
			peak = Math.max(peak, p)
		}
		v += h * a
		z += h * v
		if (!Number.isFinite(z) || Math.abs(z) > 1e4) z = Math.sign(z || 1) * 1e4
		if (touched) rebound = Math.max(rebound, z)
		zs.push(z)
	}
	return { zs, tc, peak, rebound, hw: h / (tc * dampratio) }
}

export default function (root: HTMLElement, lang: Lang) {
	const L = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const num = (x: number, d: number) => x.toFixed(d).replace('.', L('.', ','))
	const f = frame(root, lang, {
		title: { en: 'The contact is a spring', tr: 'Contact bir yaydır' },
		caption: {
			en: 'A 1 kg body falls 10 cm onto the floor at dt = 2 ms. timeconst sets how soft the contact spring is, dampratio how much it bounces. Turn refsafe off and push timeconst below 2·dt, or keep refsafe on and lower dampratio: both make h·ω large, and the contact starts adding energy.',
			tr: '1 kg’lık bir cisim dt = 2 ms’de zemine 10 cm’den düşüyor. timeconst contact yayının ne kadar yumuşak olduğunu, dampratio ne kadar sektirdiğini belirler. refsafe’i kapatıp timeconst’u 2·dt’nin altına indir ya da refsafe açıkken dampratio’yu düşür: ikisi de h·ω’yı büyütür ve contact enerji eklemeye başlar.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 300)
	const tcS = f.slider({ en: 'solref timeconst', tr: 'solref timeconst' }, { min: 0, max: TCS.length - 1, step: 1, value: 7, fmt: (i) => `${num(TCS[i], 4).replace(/0+$/, '')} s` })
	const drS = f.slider({ en: 'solref dampratio', tr: 'solref dampratio' }, { min: 0, max: DRS.length - 1, step: 1, value: 7, fmt: (i) => num(DRS[i], 2) })
	const safe = f.checkbox({ en: 'refsafe (timeconst ≥ 2·dt)', tr: 'refsafe (timeconst ≥ 2·dt)' }, true)
	const rTc = f.readout({ en: 'timeconst used', tr: 'kullanılan timeconst' })
	const rHw = f.readout({ en: 'h·ω', tr: 'h·ω' })
	const rPen = f.readout({ en: 'deepest penetration', tr: 'en derin penetration' })
	const rReb = f.readout({ en: 'rebound', tr: 'sekme' })
	const note = f.note({ en: '', tr: '' })

	let run = simulate(TCS[7], DRS[7], true), tp = 0
	const restart = () => { run = simulate(TCS[tcS.get()], DRS[drS.get()], safe.get()); tp = 0 }
	for (const c of [tcS, drS]) c.on(restart)
	safe.on(restart)
	f.button({ en: 'Drop again', tr: 'Yeniden bırak' }, () => { tp = 0 })

	animate(root, (dt) => {
		tp = Math.min(T_END, tp + dt * 0.35)
		const idx = Math.min(run.zs.length - 1, Math.floor(tp / h))
		const c = colors()
		ctx.clearRect(0, 0, W, H)
		ctx.font = mono

		// left: the body, 1 mm = 2 px vertically so the penetration is visible
		const floorY = 250, S = 2000, bx = 50, bw = 70
		const Yb = (z: number) => floorY - Math.max(-0.03, Math.min(0.11, z)) * S
		const z = run.zs[idx]
		ctx.fillStyle = c.line; ctx.fillRect(16, floorY, 150, H - floorY - 20)
		ctx.strokeStyle = c.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(16, floorY); ctx.lineTo(166, floorY); ctx.stroke()
		ctx.fillStyle = c.accent; ctx.fillRect(bx, Yb(z) - bw, bw, bw)
		if (z < 0) { ctx.fillStyle = c.ink; ctx.fillRect(bx, floorY, bw, Yb(z) - floorY) }
		ctx.fillStyle = c.muted
		ctx.fillText(L('floor', 'zemin'), 18, floorY + 18)
		ctx.fillText(z > 0.11 ? `↑ ${num(z, 2)} m` : '', bx, 28)
		ctx.fillText(L('1 mm = 2 px', '1 mm = 2 px'), 18, H - 6)

		// right: height of the bottom face over time
		const a = axes(ctx, { x0: 220, y0: 18, x1: W - 16, y1: H - 34 }, [0, T_END], [-15, 110])
		a.draw(c, L('time [s] →', 'zaman [s] →'), L('bottom face [mm]', 'alt yüz [mm]'))
		a.line([[0, 0], [T_END, 0]], c.line, 1.5, [5, 5])
		a.line([[0, DROP * 1000], [T_END, DROP * 1000]], c.line, 1, [2, 4])
		ctx.save(); ctx.beginPath(); ctx.rect(220, 10, W - 236, H - 44); ctx.clip()
		const pts: [number, number][] = []
		for (let i = 0; i <= idx; i++) pts.push([i * h, Math.max(-30, Math.min(150, run.zs[i] * 1000))])
		a.line(pts, c.accent)
		ctx.restore()
		ctx.fillStyle = c.muted
		ctx.fillText(L('dotted = drop height', 'noktalı = bırakma yüksekliği'), 300, 30)

		rTc(`${num(run.tc, 4)} s`)
		rHw(num(run.hw, 2))
		rPen(`${num(run.peak * 1000, 2)} mm`)
		const done = tp >= T_END
		rReb(done ? (run.rebound > 1 ? `${num(run.rebound, 2)} m` : `${num(run.rebound * 1000, 1)} mm`) : '…')
		if (!done) { note(''); return }
		const gained = run.rebound > DROP * 1.001
		if (gained && !safe.get() && TCS[tcS.get()] < 2 * h) note(L('timeconst is below 2·dt with the guard off: the spring is too stiff for the step and the contact injects energy.', 'Koruma kapalıyken timeconst 2·dt’nin altında: yay bu adım için fazla sert ve contact enerji pompalıyor.'))
		else if (gained) note(L('The guard is on, but ω = 1/(timeconst·dampratio) is still large: the body comes back higher than it was dropped from.', 'Koruma açık, ama ω = 1/(timeconst·dampratio) yine büyük: cisim bırakıldığı yükseklikten daha yükseğe dönüyor.'))
		else if (run.rebound > 0.0005) note(L('dampratio below 1: the contact bounces.', 'dampratio 1’in altında: contact sektiriyor.'))
		else note(L('No bounce. A larger timeconst sinks deeper on impact; twice the timeconst rests about four times as deep.', 'Sekme yok. Daha büyük timeconst çarpmada daha derine batar; iki kat timeconst dururken yaklaşık dört kat derine iner.'))
	})
}
