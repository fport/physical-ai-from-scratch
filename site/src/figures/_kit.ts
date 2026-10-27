// Shared helpers for interactive lesson figures. A figure module (site/src/figures/NN-name.ts)
// exports `default (root, lang) => void`; it calls frame() to build the standard box (tag, title,
// caption, toy-model note, canvas, controls, readouts) and then draws on the canvas.
// Figures are small JavaScript toy models, never MuJoCo; frame() always prints that note.
export type Lang = 'en' | 'tr'
export type Text = { en: string; tr: string }
export type Colors = { ink: string; muted: string; line: string; accent: string; soft: string; bg: string; panel: string }

export const mono = '13px JetBrains Mono, monospace'
export const tr = (lang: Lang, s: Text) => s[lang]

export const colors = (): Colors => {
	const s = getComputedStyle(document.documentElement)
	const v = (n: string) => s.getPropertyValue(n).trim() || '#000'
	return {
		ink: v('--fa-text'), muted: v('--fa-muted'), line: v('--fa-line'), accent: v('--fa-accent'),
		soft: v('--fa-accent-soft'), bg: v('--fa-bg'), panel: v('--fa-panel'),
	}
}

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string) => {
	const e = document.createElement(tag)
	if (cls) e.className = cls
	if (text) e.textContent = text
	return e
}

export interface Frame {
	lang: Lang
	/** canvas with a CSS-pixel drawing size W×H (scaled for the device pixel ratio) */
	canvas(W?: number, H?: number): { ctx: CanvasRenderingContext2D; W: number; H: number; cv: HTMLCanvasElement }
	slider(label: Text, o: { min: number; max: number; step: number; value: number; fmt?: (x: number) => string }): Control<number>
	checkbox(label: Text, value?: boolean): Control<boolean>
	select(label: Text, options: Text[], value?: number): Control<number>
	button(label: Text, onClick: () => void): HTMLButtonElement
	readout(label: Text): (value: string) => void
	note(text: Text): (value?: Text | string) => void
}

export interface Control<T> {
	get(): T
	set(v: T): void
	on(fn: (v: T) => void): void
}

export function frame(root: HTMLElement, lang: Lang, o: { title: Text; caption: Text }): Frame {
	root.replaceChildren()
	const head = el('div', 'fa-fig-head')
	head.append(
		el('p', 'fa-fig-tag', lang === 'tr' ? 'ŞEKİL · ETKİLEŞİMLİ' : 'FIGURE · INTERACTIVE'),
		el('p', 'fa-fig-title', o.title[lang]),
		el('p', 'fa-fig-caption', o.caption[lang]),
	)
	const stage = el('div', 'fa-fig-stage')
	const controls = el('div', 'fa-fig-controls')
	const readouts = el('div', 'fa-fig-readout')
	const toy = el('p', 'fa-fig-toy', lang === 'tr'
		? 'Tarayıcıda oyuncak model, MuJoCo değil. Ölçülmüş sonuçlar bu dersin Deney bölümünde.'
		: 'Browser toy model, not MuJoCo. Measured results are in this lesson’s Experiment section.')
	root.append(head, stage, controls, readouts, toy)

	const control = <T>(input: HTMLInputElement | HTMLSelectElement, read: () => T, write: (v: T) => void, update?: () => void): Control<T> => {
		const fns: ((v: T) => void)[] = []
		input.addEventListener('input', () => { update?.(); fns.forEach((f) => f(read())) })
		return { get: read, set: (v) => { write(v); update?.(); fns.forEach((f) => f(read())) }, on: (f) => fns.push(f) }
	}

	return {
		lang,
		canvas(W = 640, H = 300) {
			const cv = el('canvas')
			const dpr = Math.min(2, window.devicePixelRatio || 1)
			cv.width = W * dpr; cv.height = H * dpr
			cv.style.aspectRatio = `${W} / ${H}`
			stage.append(cv)
			const ctx = cv.getContext('2d')!
			ctx.scale(dpr, dpr)
			return { ctx, W, H, cv }
		},
		slider(label, s) {
			const wrap = el('label'), out = el('output'), inp = el('input')
			inp.type = 'range'; inp.min = String(s.min); inp.max = String(s.max); inp.step = String(s.step); inp.value = String(s.value)
			const fmt = s.fmt || ((x: number) => String(x))
			const upd = () => (out.textContent = fmt(Number(inp.value)))
			wrap.append(el('span', '', label[lang] + ' '), out, inp)
			controls.append(wrap); upd()
			return control(inp, () => Number(inp.value), (v) => (inp.value = String(v)), upd)
		},
		checkbox(label, value = false) {
			const wrap = el('label', 'fa-check'), inp = el('input')
			inp.type = 'checkbox'; inp.checked = value
			wrap.append(inp, document.createTextNode(' ' + label[lang]))
			controls.append(wrap)
			return control(inp, () => inp.checked, (v) => (inp.checked = v))
		},
		select(label, options, value = 0) {
			const wrap = el('label'), sel = el('select')
			options.forEach((t, i) => { const op = el('option', '', t[lang]); op.value = String(i); sel.append(op) })
			sel.value = String(value)
			wrap.append(el('span', '', label[lang]), sel)
			controls.append(wrap)
			return control(sel, () => Number(sel.value), (v) => (sel.value = String(v)))
		},
		button(label, onClick) {
			const b = el('button', '', label[lang])
			b.type = 'button'; b.addEventListener('click', onClick)
			controls.append(b)
			return b
		},
		readout(label) {
			const span = el('span', '', label[lang] + ' '), b = el('b')
			span.append(b); readouts.append(span)
			return (v) => (b.textContent = v)
		},
		note(text) {
			const p = el('p', 'fa-fig-note', text[lang])
			readouts.after(p)
			return (v) => (p.textContent = v === undefined ? text[lang] : typeof v === 'string' ? v : v[lang])
		},
	}
}

/** requestAnimationFrame loop that only runs while the figure is on screen. dt in seconds (capped). */
export function animate(root: HTMLElement, step: (dt: number, t: number) => void) {
	let visible = false, last = 0, t = 0, raf = 0
	const tick = (now: number) => {
		const dt = last ? Math.min(0.05, (now - last) / 1000) : 0
		last = now; t += dt
		step(dt, t)
		raf = visible ? requestAnimationFrame(tick) : 0
	}
	new IntersectionObserver(([e]) => {
		visible = e.isIntersecting
		if (visible && !raf) { last = 0; raf = requestAnimationFrame(tick) }
	}).observe(root)
}

/** Seeded random numbers so a figure looks the same on every load. */
export function rng(seed = 1) {
	let s = seed % 2147483647 || 1
	const u = () => (s = (s * 16807) % 2147483647) / 2147483647
	return { u, gauss: () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u()) }
}

/** Axes helper: maps data ranges to a canvas box and draws a frame with labels. */
export function axes(ctx: CanvasRenderingContext2D, box: { x0: number; y0: number; x1: number; y1: number }, xr: [number, number], yr: [number, number]) {
	const X = (v: number) => box.x0 + ((v - xr[0]) / (xr[1] - xr[0])) * (box.x1 - box.x0)
	const Y = (v: number) => box.y1 - ((v - yr[0]) / (yr[1] - yr[0])) * (box.y1 - box.y0)
	const draw = (c: Colors, xl = '', yl = '') => {
		ctx.strokeStyle = c.line; ctx.lineWidth = 1.5
		ctx.beginPath(); ctx.moveTo(box.x0, box.y0); ctx.lineTo(box.x0, box.y1); ctx.lineTo(box.x1, box.y1); ctx.stroke()
		ctx.font = mono
		// axis names sit inside the plot on a background patch, so tick numbers below the axis stay free
		const label = (s: string, x: number, y: number) => {
			const w = ctx.measureText(s).width
			ctx.fillStyle = c.bg; ctx.globalAlpha = 0.85; ctx.fillRect(x - 3, y - 11, w + 6, 15); ctx.globalAlpha = 1
			ctx.fillStyle = c.muted; ctx.fillText(s, x, y)
		}
		if (yl) label(yl, box.x0 + 6, box.y0 + 12)
		if (xl) label(xl, box.x1 - ctx.measureText(xl).width - 4, box.y1 - 6)
	}
	const line = (pts: [number, number][], color: string, width = 2.5, dash: number[] = []) => {
		ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.beginPath()
		pts.forEach(([a, b], i) => (i ? ctx.lineTo(X(a), Y(b)) : ctx.moveTo(X(a), Y(b))))
		ctx.stroke(); ctx.setLineDash([])
	}
	return { X, Y, draw, line }
}
