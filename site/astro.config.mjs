// @ts-check
import { defineConfig } from 'astro/config'
import starlight from '@astrojs/starlight'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import sidebar from './src/sidebar.generated.json' with { type: 'json' }

const REPO = 'https://github.com/fport/physical-ai-from-scratch'

export default defineConfig({
  site: process.env.SITE_URL,
  base: process.env.SITE_BASE || '/',
  markdown: { remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] },
  integrations: [
    starlight({
      title: { en: 'Physical AI', tr: 'Fiziksel AI' },
      description: 'Physical AI Engineering from Scratch: an open course book with runnable MuJoCo experiments, from simulation to a real SO-101 arm.',
      defaultLocale: 'root',
      locales: {
        root: { label: 'English', lang: 'en' },
        tr: { label: 'Türkçe', lang: 'tr' },
      },
      social: [{ icon: 'github', label: 'GitHub', href: REPO }],
      sidebar,
      customCss: [
        '@fontsource-variable/inter',
        '@fontsource/jetbrains-mono/400.css',
        '@fontsource/jetbrains-mono/700.css',
        '@fontsource/silkscreen/400.css',
        'katex/dist/katex.min.css',
        './src/styles/custom.css',
      ],
      components: {
        ThemeProvider: './src/components/ThemeProvider.astro',
        Sidebar: './src/components/Sidebar.astro',
      },
      expressiveCode: {
        themes: ['github-dark', 'github-light'],
        styleOverrides: { borderRadius: '0.5rem', frames: { shadowColor: 'transparent' } },
      },
      lastUpdated: false,
      pagination: true,
      tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
    }),
  ],
})
