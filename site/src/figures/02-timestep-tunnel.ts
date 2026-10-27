// Lesson 02 · the lesson's drop scene: a 5 cm cube released at 1.30 m (centre) onto a 1 cm tabletop
// whose top is at 30 cm (mid-plane 29.5 cm). Semi-implicit Euler at the chosen dt, collision checked
// only at the step instants, and one soft contact in the style of MuJoCo's solref/solimp that pushes
// out through the nearer face (up above the mid-plane, down below it). The reference spring is 10 %
// stiffer than a single MuJoCo contact; with that the toy lands on the same stayed/through side as
// every row of the lesson's sweep. It is still a toy, not the measurement.
import { animate, colors, frame, mono, type Lang } from './_kit'

const DTS = [0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50] // ms, the lesson's sweep
const g = 9.81, half = 0.025, top = 0.3, bot = 0.29, mid = 0.295, z0 = 1.3, T_END = 0.8

type Run = { h: number; z: number[]; firstOverlap: number; through: boolean }

// solimp default 0.9 0.95 0.001 0.5 2: impedance rises from 0.9 to 0.95 over the first millimetre
const impedance = (p: number) => {
	const x = Math.min(1, Math.max(0, p / 0.001))
	const y = x < 0.5 ? 0.5 * (x / 0.5) ** 2 : 1 - 0.5 * ((1 - x) / 0.5) ** 2
	return 0.9 + 0.05 * y
}

function simulate(dtMs: number, timeconst: number): Run {
	const h = dtMs / 1000, tc = Math.max(timeconst, 2 * h) // refsafe: timeconst never below 2·dt
	let z = z0, v = 0, firstOverlap = -1, through = false
	const zs: number[] = [z]
	for (let t = 0; t < T_END; t += h) {
		let a = -g
		if (z - half < top && z + half > bot) {
			if (firstOverlap < 0) firstOverlap = zs.length - 1
			const up = z > mid, n = up ? 1 : -1
			const p = up ? top - (z - half) : z + half - bot // depth through the nearer face
			const d = impedance(p), dw = 0.95
			const b = 2 / (dw * tc), k = (1.1 * d) / (dw * dw * tc * tc)
			const au = -g * n, aref = -b * v * n + k * p
			a = Math.max(au, (1 - d) * au + d * aref) * n // a contact can push, never pull
		}
		v += h * a
		z += h * v // move with the new velocity
		if (z < half) { z = half; v = 0 } // the floor
		if (z < bot - half) through = true
		zs.push(z)
	}
	return { h, z: zs, firstOverlap, through }
}

export default function (root: HTMLElement, lang: Lang) {
	const L = (en: string, tr: string) => (lang === 'tr' ? tr : en)
	const f = frame(root, lang, {
		title: { en: 'A strobe, not a film', tr: 'Film değil, çakar ışık' },
		caption: {
			en: 'The lesson’s scene: a 5 cm cube falls 97.5 cm onto a 1 cm tabletop. The right panel is a zoom around the tabletop: one bar per timestep, showing where the cube is at that instant. Contact exists only where a bar overlaps the slab. Pick a dt and watch which side of the dashed mid-plane the first overlapping bar lands on.',
			tr: 'Dersteki sahne: 5 cm’lik bir küp 97,5 cm düşüp 1 cm’lik bir tablaya çarpıyor. Sağdaki panel tablanın çevresine yakınlaştırma: her timestep için bir çubuk, küpün o andaki yerini gösteriyor. Contact yalnızca bir çubuğun tablayla çakıştığı yerde var. Bir dt seç ve ilk çakışan çubuğun kesikli orta düzlemin hangi tarafına düştüğünü izle.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 320)
	const dtSel = f.slider({ en: 'timestep dt', tr: 'timestep dt' }, { min: 0, max: DTS.length - 1, step: 1, value: 1, fmt: (i) => `${String(DTS[i]).replace('.', L('.', ','))} ms` })
	const contact = f.select({ en: 'contact', tr: 'contact' }, [
		{ en: 'default solref 0.02 1', tr: 'default solref 0.02 1' },
		{ en: 'stiff solref 0.005 1', tr: 'sert solref 0.005 1' },
	])
	const rJump = f.readout({ en: 'jump per step', tr: 'adım başına sıçrama' })
	const rFirst = f.readout({ en: 'centre at 1st overlap', tr: 'ilk çakışmada merkez' })
	const rSeen = f.readout({ en: 'contact seen', tr: 'contact görüldü' })
	const rRes = f.readout({ en: 'result', tr: 'sonuç' })
	const note = f.note({ en: '', tr: '' })

	let run = simulate(DTS[1], 0.02), tp = 0
	const restart = () => { run = simulate(DTS[dtSel.get()], contact.get() ? 0.005 : 0.02); tp = 0 }
	f.button({ en: 'Drop again', tr: 'Yeniden bırak' }, () => { tp = 0 })
	dtSel.on(restart)
	contact.on(restart)

	const cm = (m: number) => (m * 100).toFixed(2).replace('.', L('.', ','))
	const T0 = 0.41, T1 = 0.49 // zoom time window [s]
	const ZLO = 0.215, ZHI = 0.345 // zoom height window [m]

	animate(root, (dt) => {
		tp = Math.min(T_END, tp + dt * (tp < T0 ? 0.6 : tp < T1 ? 0.04 : 0.6))
		const c = colors()
		ctx.clearRect(0, 0, W, H)
		ctx.font = mono
		const idx = Math.min(run.z.length - 1, Math.floor(tp / run.h))

		// left: the whole drop
		const Yl = (z: number) => H - 24 - z * 205
		ctx.fillStyle = c.line; ctx.fillRect(14, Yl(0), 120, 2)
		ctx.fillStyle = c.ink; ctx.fillRect(40, Yl(top), 70, 3)
		ctx.strokeStyle = c.muted; ctx.lineWidth = 1; ctx.setLineDash([3, 3])
		ctx.strokeRect(30, Yl(ZHI), 90, Yl(ZLO) - Yl(ZHI)); ctx.setLineDash([])
		ctx.fillStyle = c.accent
		const zl = run.z[idx]
		ctx.fillRect(75 - 5, Yl(zl) - 5, 10, 10)
		ctx.fillStyle = c.muted
		ctx.fillText('1.30 m'.replace('.', L('.', ',')), 16, Yl(z0) + 4)
		ctx.fillText(L('zoom', 'yakın'), 124, Yl(ZHI) + 12)

		// right: zoom, time on x, height on y, one bar per step
		const x0 = 190, x1 = W - 16, y0 = 16, y1 = H - 30
		const X = (t: number) => x0 + ((t - T0) / (T1 - T0)) * (x1 - x0)
		const Y = (z: number) => y1 - ((z - ZLO) / (ZHI - ZLO)) * (y1 - y0)
		ctx.fillStyle = c.soft; ctx.fillRect(x0, Y(top), x1 - x0, Y(bot) - Y(top))
		ctx.strokeStyle = c.ink; ctx.lineWidth = 1.5; ctx.strokeRect(x0, Y(top), x1 - x0, Y(bot) - Y(top))
		ctx.strokeStyle = c.muted; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(x0, Y(mid)); ctx.lineTo(x1, Y(mid)); ctx.stroke(); ctx.setLineDash([])
		ctx.fillStyle = c.muted
		ctx.fillText(L('tabletop 1 cm', 'tabla 1 cm'), x0 + 6, Y(top) - 6)
		ctx.fillText(L('dashed: mid-plane 29.5 cm', 'kesikli: orta düzlem 29,5 cm'), x1 - 230, Y(bot) + 16)
		ctx.fillText(L('time →', 'zaman →'), x1 - 58, H - 10)
		ctx.save(); ctx.beginPath(); ctx.rect(x0, y0 - 6, x1 - x0, y1 - y0 + 12); ctx.clip()
		const first = Math.max(0, Math.floor(T0 / run.h) - 1)
		for (let i = first; i <= idx; i++) {
			const t = i * run.h
			if (t > T1) break
			const z = run.z[i], x = X(t)
			const overlap = z - half < top && z + half > bot
			const isFirst = i === run.firstOverlap
			ctx.strokeStyle = isFirst ? c.accent : overlap ? c.ink : c.muted
			ctx.lineWidth = isFirst ? 4 : overlap ? 2 : 1.2
			ctx.globalAlpha = overlap ? 1 : 0.55
			ctx.beginPath(); ctx.moveTo(x, Y(z + half)); ctx.lineTo(x, Y(z - half)); ctx.stroke()
			ctx.globalAlpha = 1
			ctx.fillStyle = c.accent; ctx.beginPath(); ctx.arc(x, Y(z), overlap ? 3.5 : 2.2, 0, 7); ctx.fill()
			if (overlap) { // direction of the contact push on the cube
				const dir = z > mid ? -1 : 1, yy = Y(z) + dir * 12
				ctx.fillStyle = z > mid ? c.ink : c.accent
				ctx.beginPath(); ctx.moveTo(x, yy + dir * 7); ctx.lineTo(x - 4, yy); ctx.lineTo(x + 4, yy); ctx.fill()
			}
		}
		ctx.restore()
		ctx.fillStyle = c.muted
		ctx.fillText(L('bar = cube at one step · ▲▼ = contact push', 'çubuk = bir adımda küp · ▲▼ = contact itişi'), x0, H - 10)

		// readouts
		const h = run.h, vImp = Math.sqrt(2 * g * (z0 - half - top))
		rJump(`${cm(vImp * h)} cm`)
		const done = tp >= T1
		const seen = run.firstOverlap >= 0
		const zf = seen ? run.z[run.firstOverlap] : 0
		rFirst(seen && tp >= run.firstOverlap * h ? `${cm(zf)} cm` + (zf < mid ? L(' (below)', ' (altında)') : L(' (above)', ' (üstünde)')) : '—')
		rSeen(done || (seen && tp >= run.firstOverlap * h) ? (seen ? L('yes', 'evet') : L('no', 'hayır')) : '…')
		rRes(done ? (run.through ? L('fell through', 'geçti') : L('stayed', 'kaldı')) : '…')
		if (!done) note('')
		else if (!seen) note(L('Missed contact: no step lands in the overlap zone, so the engine never sees the table.', 'Kaçırılan contact: hiçbir adım çakışma bölgesine düşmüyor, engine masayı hiç görmüyor.'))
		else if (zf < mid) note(L('Wrong-side overlap: the first overlapping step is already below the mid-plane, so the contact pushes the cube down. A stiffer contact cannot help.', 'Yanlış taraftan çakışma: ilk çakışan adım zaten orta düzlemin altında, contact küpü aşağı itiyor. Daha sert bir contact işe yaramaz.'))
		else if (run.through) note(L('Too-soft contact: the first touch is on the right side, but the cube sinks past the mid-plane and the push flips down. Try the stiff contact.', 'Fazla yumuşak contact: ilk temas doğru tarafta, ama küp orta düzlemi geçecek kadar batıyor ve itiş aşağı dönüyor. Sert contact’ı dene.'))
		else note(L('Stayed. Move dt one notch either way: the outcome is not monotonic in dt.', 'Kaldı. dt’yi bir kademe oynat: sonuç dt’ye göre monoton değil.'))
	})
}
