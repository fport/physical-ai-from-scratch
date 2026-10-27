// Lesson 31 · action chunks and a pause the observation cannot time. A point goes from the start to
// the target, waits 25 steps (the lesson's 0.5 s at 50 Hz), then goes to the drop point. The policy
// sees only its position and the target, no timer. Its chunk is what the demonstrations did next from
// that observation; while resting on the target that is ambiguous (the demo could be at any pause
// step), so each chunk entry is the per-entry median over the pause steps, as an L1 loss would give.
// Execution is open-loop (play the chunk, then query) or temporal ensembling with w = exp(−m·age),
// m = 0.01, older predictions weighing more, as in the lesson's code. Actions are absolute position
// targets and the point follows them exactly; there is no noise and no network.
import { animate, colors, frame, mono, type Lang } from './_kit'

type P2 = [number, number]

export default function (root: HTMLElement, lang: Lang) {
	const f = frame(root, lang, {
		title: { en: 'Can the chunk reach past the pause?', tr: 'Chunk beklemenin ötesine uzanabiliyor mu?' },
		caption: {
			en: 'Dots: the chunk the policy just predicted. At rest on the target, short chunks are all “hold”, so the point never leaves. A chunk long enough to contain the departure leaves; ensembling can average that departure away. Move the target to see how fast each scheme reacts. Shown at half speed.',
			tr: 'Noktalar: policy’nin az önce tahmin ettiği chunk. Hedefin üstünde dururken kısa chunk’ların hepsi “bekle”dir, nokta hiç ayrılmaz. Ayrılışı içerecek kadar uzun bir chunk ayrılır; ensembling bu ayrılışı ortalamada eritebilir. Her şemanın ne kadar hızlı tepki verdiğini görmek için hedefi kaydır. Yarı hızda gösteriliyor.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 300)
	const KS = [1, 4, 8, 16, 32]
	const kSel = f.select({ en: 'chunk size k', tr: 'chunk boyu k' }, KS.map((k) => ({ en: String(k), tr: String(k) })), 2)
	const scheme = f.select({ en: 'execution', tr: 'yürütme' }, [{ en: 'open-loop', tr: 'open-loop' }, { en: 'temporal ensembling', tr: 'temporal ensembling' }], 0)
	const rAt = f.readout({ en: 'time on the target', tr: 'hedefte geçen süre' })
	const rStat = f.readout({ en: 'status', tr: 'durum' })
	const num = (x: number, d: number) => (lang === 'tr' ? x.toFixed(d).replace('.', ',') : x.toFixed(d))

	const PAUSE = 25, V = 0.03, TOL = 0.04, HZ = 50, BUDGET = 200
	const START: P2 = [-1, -0.4], DROP: P2 = [1, 0.5]
	let target: P2 = [0, 0]
	const sub = (a: P2, b: P2): P2 => [a[0] - b[0], a[1] - b[1]]
	const add = (a: P2, b: P2): P2 => [a[0] + b[0], a[1] + b[1]]
	const mul = (a: P2, s: number): P2 => [a[0] * s, a[1] * s]
	const nrm = (a: P2) => Math.hypot(a[0], a[1])
	const toward = (p: P2, q: P2, d: number): P2 => { const e = sub(q, p), n = nrm(e); return n <= d ? [q[0], q[1]] : add(p, mul(e, d / n)) }

	/** what the demonstrations did in the next k steps from position p */
	const chunk = (p: P2, k: number): P2[] => {
		const out: P2[] = [], T = target
		const u = mul(sub(DROP, T), 1 / nrm(sub(DROP, T))), r = sub(p, T)
		const s = r[0] * u[0] + r[1] * u[1], lat = nrm(sub(r, mul(u, s)))
		if (nrm(r) <= TOL) {
			// resting on the target: demo pause step j is unknown (uniform over 0..PAUSE-1); entry i has
			// moved max(0, j + i - PAUSE + 1) steps; the median over j is max(0, i - (PAUSE - 1) / 2)
			for (let i = 0; i < k; i++) out.push(toward(T, DROP, Math.max(0, i - (PAUSE - 1) / 2) * V))
			return out
		}
		if (s > 0 && lat < 0.08) { // already on the way to the drop point
			let q = p
			for (let i = 0; i < k; i++) { q = toward(q, DROP, V); out.push(q) }
			return out
		}
		let q = p, arrive = -1 // approaching: reach the target, hold PAUSE steps, then leave
		for (let i = 0; i < k; i++) {
			if (arrive < 0) { q = toward(q, T, V); if (nrm(sub(q, T)) <= TOL) arrive = i; out.push(q) }
			else { const n = i - arrive - PAUSE; out.push(n <= 0 ? q : toward(q, DROP, n * V)) }
		}
		return out
	}

	let pos: P2 = [...START], step = 0, cur: P2[] = [], shown: P2[] = [], cmd: P2 = [...START]
	let num_: P2[] = [], den: number[] = [], trail: P2[] = [], onT = 0, phase = 0, endAt = -1, acc = 0
	const restart = () => {
		target = [0, 0]; pos = [...START]; cmd = [...START]; step = 0; cur = []; shown = []
		num_ = []; den = []; trail = []; onT = 0; phase = 0; endAt = -1; acc = 0
	}
	restart()
	kSel.on(restart); scheme.on(restart)
	f.button({ en: 'Move the target', tr: 'Hedefi kaydır' }, () => {
		const a = (step * 2.399) % (2 * Math.PI)
		target = [0.3 * Math.cos(a), 0.22 * Math.sin(a)]
		if (endAt < 0) { phase = 0; onT = 0 }
	})
	f.button({ en: 'Restart', tr: 'Yeniden başlat' }, restart)

	const controlStep = () => {
		const k = KS[kSel.get()]
		if (scheme.get() === 0) {
			if (step % k === 0) cur = chunk(pos, k)
			cmd = cur[step % k]
			shown = cur.slice(step % k)
		} else {
			const ch = chunk(pos, k)
			for (let j = 0; j < k; j++) {
				const w = Math.exp(-0.01 * (k - 1 - j)), t = step + j
				num_[t] = add(num_[t] || [0, 0], mul(ch[j], w)); den[t] = (den[t] || 0) + w
			}
			cmd = mul(num_[step], 1 / den[step])
			shown = ch
		}
		pos = cmd
		trail.push(pos)
		step++
		const dT = nrm(sub(pos, target))
		if (dT <= 0.05) { onT++; if (phase === 0) phase = 1 } else { if (phase === 1) phase = 2; onT = 0 }
		if (nrm(sub(pos, DROP)) <= 0.03 && phase === 2) phase = 3
		if (phase === 3 || step >= BUDGET) endAt = step
	}

	const S = 230
	const X = (x: number) => W / 2 + x * S, Y = (y: number) => 170 - y * S
	animate(root, (dt) => {
		acc += dt * HZ * 0.5 // half speed
		while (acc >= 1) {
			acc -= 1
			if (endAt < 0) controlStep()
			else if (step - endAt > 40) restart()
			else step++
		}
		const c = colors()
		ctx.clearRect(0, 0, W, H)
		ctx.font = mono
		// start, target (with tolerance), drop
		ctx.fillStyle = c.muted; ctx.beginPath(); ctx.arc(X(START[0]), Y(START[1]), 5, 0, 7); ctx.fill()
		ctx.fillText(lang === 'tr' ? 'başlangıç' : 'start', X(START[0]) - 30, Y(START[1]) + 22)
		ctx.strokeStyle = c.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(target[0]), Y(target[1]), 0.05 * S, 0, 7); ctx.stroke()
		ctx.fillStyle = c.muted; ctx.fillText(lang === 'tr' ? 'hedef: 25 adım bekle' : 'target: wait 25 steps', X(target[0]) - 70, Y(target[1]) + 32)
		ctx.strokeStyle = c.ink; ctx.strokeRect(X(DROP[0]) - 8, Y(DROP[1]) - 8, 16, 16)
		ctx.fillText(lang === 'tr' ? 'bırakma' : 'drop', X(DROP[0]) - 18, Y(DROP[1]) - 14)
		// trail, chunk, point
		ctx.strokeStyle = c.line; ctx.lineWidth = 2; ctx.beginPath()
		trail.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))))
		ctx.stroke()
		ctx.fillStyle = c.accent
		for (const q of shown) { ctx.beginPath(); ctx.arc(X(q[0]), Y(q[1]), 3, 0, 7); ctx.fill() }
		ctx.fillStyle = c.ink; ctx.beginPath(); ctx.arc(X(pos[0]), Y(pos[1]), 7, 0, 7); ctx.fill()
		ctx.fillStyle = c.muted
		ctx.fillText(lang === 'tr' ? `adım ${Math.min(step, BUDGET)}/${BUDGET}` : `step ${Math.min(step, BUDGET)}/${BUDGET}`, 14, 20)
		rAt(num(onT / HZ, 2) + ' s')
		const stuck = phase === 1 && onT > PAUSE + 50
		rStat(phase === 3 ? (lang === 'tr' ? 'bırakma noktasına ulaştı' : 'reached the drop point')
			: endAt >= 0 ? (phase === 1 ? (lang === 'tr' ? 'hedefte takılı kaldı' : 'stuck on the target') : (lang === 'tr' ? 'süre bitti' : 'out of time'))
			: stuck ? (lang === 'tr' ? 'takılı mı? hâlâ hedefte' : 'stuck? still on the target')
			: phase === 0 ? (lang === 'tr' ? 'hedefe gidiyor' : 'approaching the target')
			: phase === 1 ? (lang === 'tr' ? 'hedefte bekliyor' : 'waiting on the target')
			: (lang === 'tr' ? 'bırakma noktasına gidiyor' : 'heading to the drop point'))
	})
}
