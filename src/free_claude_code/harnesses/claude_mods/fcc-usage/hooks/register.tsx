import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register, Timer } from 'claude-code'

import type { Period, Report } from '../types'

const PANE = 'fcc-usage'
const TITLE = 'fcc Usage'
const FALLBACK_URL = 'http://127.0.0.1:8082'
const REFRESH_MS = 30_000

const report = atom({ plugin: 'fcc-usage', key: 'report' } as const, {
  text: null,
  error: null,
  period: 'day',
  loading: false,
  updatedAt: null,
} as Report)

let timer: Timer | null = null

async function load($: EngineInterface, period?: Period): Promise<void> {
  const current = await read($, report)
  const wanted = period ?? current.period
  await update($, report, r => ({ ...r, period: wanted, loading: true }))
  const base = ((await $.env.get('ANTHROPIC_BASE_URL')) || FALLBACK_URL).replace(/\/+$/, '')
  const session = await $.session.id()
  const url = `${base}/api/usage/text?period=${wanted}&session=${encodeURIComponent(session)}`
  try {
    const res = await $.http.fetch(url)
    if (!res.ok) throw new Error(`fcc answered HTTP ${res.status}`)
    const now = await $.clock.now()
    await update($, report, r => ({ ...r, text: res.text, error: null, loading: false, updatedAt: now }))
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    await update($, report, r => ({
      ...r,
      error: `fcc is not reachable at ${base} (${message})`,
      loading: false,
    }))
  }
}

async function open($: EngineInterface, period?: Period): Promise<void> {
  // focus + closeOnEscape + holdToasts: the pane behaves as a dialog, like /usage.
  await $.ui.open({ id: PANE, title: TITLE, focus: true, closeOnEscape: true, holdToasts: true })
  await load($, period)
  timer?.cancel()
  timer = $.clock.every(REFRESH_MS, () => void load($))
}


export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'fcc-usage',
      description: 'fcc Usage panel: session cost and tokens, provider limits, what drives usage (d day, w week, r refresh)',
    })
    return next(e)
  })

  on('command.run', { command: 'fcc-usage' }, async ($, e) => {
    const arg = e.args.trim().toLowerCase()
    await open($, arg === 'week' || arg === 'w' ? 'week' : arg === 'day' || arg === 'd' ? 'day' : undefined)
    return { text: 'fcc Usage panel opened.' }
  })

  on('ui.close', async ($, e, next) => {
    if (e.id === PANE) {
      timer?.cancel()
      timer = null
    }
    return next(e)
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text, Button } = $.ui.resolve(e)
    const r = await read($, report)
    const body = r.error ?? r.text ?? 'Loading…'
    const stamp = r.updatedAt ? new Date(r.updatedAt).toLocaleTimeString() : ''

    return (
      <Box flexDirection="column">
        <Box flexDirection="row" gap={1}>
          <Button hotkey="d" variant={r.period === 'day' ? 'primary' : undefined} onPress={() => void load($, 'day')}>
            Last 24h
          </Button>
          <Button hotkey="w" variant={r.period === 'week' ? 'primary' : undefined} onPress={() => void load($, 'week')}>
            Last 7 days
          </Button>
          <Button hotkey="r" onPress={() => void load($)}>
            Refresh
          </Button>
          <Text dimColor>{r.loading ? 'loading…' : stamp ? `updated ${stamp}` : ''}</Text>
        </Box>
        <Text> </Text>
        {body
          .replace(/\n+$/, '')
          .split('\n')
          .map(line => (
            <Text dimColor={/^ \S/.test(line) || / to week · /.test(line)} bold={/^(Session|Provider limits|What's contributing)/.test(line)}>
              {line || ' '}
            </Text>
          ))}
      </Box>
    )
  })
}
