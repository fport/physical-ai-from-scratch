// Lesson 29 · compounding error and DAgger on a toy lane-keeping task. State: lateral offset y and
// heading ψ; action: steering rate. The expert steers back to the centre line. The learner is a
// kernel regressor over (y, ψ) that only knows the neighbourhood of its data: far from every
// labelled state its output fades to "no steering". It also has a small steering error (bias).
// BC trains on the expert's own drives; each DAgger round lets the learner drive, has the expert
// label the states it visited, adds them to everything collected so far, and retrains.
import { animate, colors, frame, mono, rng, type Lang } from './_kit'

type S = [number, number] // y, psi
type D = [number, number, number] // y, psi, expert action

export default function (root: HTMLElement, lang: Lang) {
	const f = frame(root, lang, {
		title: { en: 'Behavior cloning drifts; DAgger pulls it back', tr: 'Behavior cloning kayar; DAgger geri çeker' },
		caption: {
			en: 'Grey: the expert’s demonstrations, all close to the centre line. Dark: the learner driving on its own. A small error takes it where no demonstration went, its next action there is a guess, and the drift compounds. Each DAgger round adds expert labels (dots) exactly where the learner went.',
			tr: 'Gri: expert’in demonstration’ları, hepsi orta çizgiye yakın. Koyu: kendi başına süren öğrenen. Küçük bir hata onu hiçbir demonstration’ın gitmediği yere götürür, oradaki bir sonraki action’ı bir tahmindir ve kayma birikir. Her DAgger turu, expert etiketlerini (noktalar) tam öğrenenin gittiği yere ekler.',
		},
	})
	const { ctx, W, H } = f.canvas(640, 300)
	const num = (x: number, d: number) => (lang === 'tr' ? x.toFixed(d).replace('.', ',') : x.toFixed(d))
	const bias = f.slider({ en: 'learner steering error', tr: 'öğrenenin direksiyon hatası' }, { min: 0, max: 0.12, step: 0.01, value: 0.08, fmt: (x) => num(x, 2) })
	const rRound = f.readout({ en: 'DAgger rounds', tr: 'DAgger turu' })
	const rData = f.readout({ en: 'labelled states', tr: 'etiketli state' })
	const rOn = f.readout({ en: 'drives that stay on the road', tr: 'yolda kalan sürüşler' })

	const T = 150, DT = 0.1, SIG = 0.02, UMAX = 0.6, ROAD = 0.5, HK = 0.06, W0 = 0.05
	const expert = (y: number, p: number) => Math.max(-UMAX, Math.min(UMAX, -(0.8 * y + 1.6 * p)))
	const drive = (pol: (y: number, p: number) => number, seed: number, b: number): S[] => {
		const r = rng(seed)
		let y = 0.02 * r.gauss(), p = 0.02 * r.gauss()
		const out: S[] = []
		for (let t = 0; t <= T; t++) {
			out.push([y, p])
			const u = pol(y, p) + b
			p += DT * u + SIG * r.gauss()
			y += DT * Math.sin(p)
		}
		return out
	}
	const label = (tr: S[]): D[] => tr.slice(0, T).map(([y, p]) => [y, p, expert(y, p)])
	// Nadaraya-Watson regression with a prior weight W0 on "no steering"
	const learner = (data: D[]) => (y: number, p: number) => {
		let sw = W0, su = 0
		for (const d of data) {
			const a = (y - d[0]) / HK, b = (p - d[1]) / HK, q = a * a + b * b
			if (q > 9) continue
			const w = Math.exp(-0.5 * q)
			sw += w; su += w * d[2]
		}
		return su / sw
	}

	const demos: S[][] = Array.from({ length: 10 }, (_, e) => drive(expert, 100 + e, 0))
	const demoData = demos.flatMap(label)
	let data: D[] = [], extra: D[] = [], round = 0, tests: S[][] = [], clock = 0
	const evaluate = () => {
		const pol = learner(data)
		tests = Array.from({ length: 6 }, (_, e) => drive(pol, 1 + e, bias.get()))
		clock = 0
	}
	const toBC = () => { data = demoData.slice(); extra = []; round = 0; evaluate() }
	const daggerRound = () => {
		const pol = learner(data)
		round++
		for (let e = 0; e < 4; e++) {
			const visited = label(drive(pol, 1000 * round + e, bias.get())) // learner drives, expert labels
			data = data.concat(visited); extra = extra.concat(visited)
		}
		evaluate()
	}
	toBC()
	f.button({ en: 'DAgger round', tr: 'DAgger turu' }, daggerRound)
	f.button({ en: 'Back to plain BC', tr: 'Düz BC’ye dön' }, toBC)
	bias.on(toBC)

	const x0 = 46, x1 = W - 14, y0 = 16, y1 = H - 30, YR = 1.2
	const X = (t: number) => x0 + (t / T) * (x1 - x0)
	const Y = (y: number) => (y0 + y1) / 2 - (y / YR) * ((y1 - y0) / 2)
	const line = (tr: S[], n: number, color: string, w: number) => {
		ctx.strokeStyle = color; ctx.lineWidth = w; ctx.beginPath()
		for (let t = 0; t < n && t < tr.length; t++) t ? ctx.lineTo(X(t), Y(tr[t][0])) : ctx.moveTo(X(t), Y(tr[t][0]))
		ctx.stroke()
	}

	animate(root, (dt) => {
		clock += dt
		const shown = Math.min(T + 1, Math.floor(clock * 60) + 1)
		const c = colors()
		ctx.clearRect(0, 0, W, H)
		ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, x1 - x0, y1 - y0); ctx.clip()
		// road
		ctx.fillStyle = c.panel; ctx.fillRect(x0, Y(ROAD), x1 - x0, Y(-ROAD) - Y(ROAD))
		ctx.strokeStyle = c.line; ctx.lineWidth = 2
		for (const e of [ROAD, -ROAD]) { ctx.beginPath(); ctx.moveTo(x0, Y(e)); ctx.lineTo(x1, Y(e)); ctx.stroke() }
		ctx.setLineDash([8, 8]); ctx.lineWidth = 1
		ctx.beginPath(); ctx.moveTo(x0, Y(0)); ctx.lineTo(x1, Y(0)); ctx.stroke(); ctx.setLineDash([])
		for (const d of demos) line(d, T + 1, c.muted, 1)
		// DAgger labels, drawn at their time step
		ctx.fillStyle = c.accent
		for (let i = 0; i < extra.length; i += 2) {
			const t = i % T
			ctx.fillRect(X(t) - 1, Y(extra[i][0]) - 1, 2.5, 2.5)
		}
		for (const tr of tests) line(tr, shown, c.ink, 2)
		ctx.restore()
		ctx.fillStyle = c.muted; ctx.font = mono
		ctx.fillText(lang === 'tr' ? 'zaman adımı →' : 'time step →', x1 - 110, H - 10)
		ctx.fillText(lang === 'tr' ? 'yanal sapma' : 'lateral offset', x0 + 4, y0 + 14)
		ctx.fillText(lang === 'tr' ? 'yol kenarı' : 'road edge', x1 - 84, Y(ROAD) - 6)
		rRound(round === 0 ? (lang === 'tr' ? '0 (düz BC)' : '0 (plain BC)') : String(round))
		rData(String(data.length))
		rOn(`${tests.filter((tr) => tr.every(([y]) => Math.abs(y) < ROAD)).length}/6`)
	})
}
