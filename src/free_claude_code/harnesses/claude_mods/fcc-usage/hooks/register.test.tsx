import { expect, test } from 'claude-code/testing'

const PANE = {
  plugin: 'fcc-usage',
  component: 'Pane',
  requestId: 'fcc-usage',
  props: {
    title: 'fcc Usage',
    isFocused: false,
    bodyColumns: 100,
    placement: 'dock',
    scroll: { offset: 0, bodyRows: 40 },
    view: {},
  },
} as const

const REPORT = 'Session\n\nTotal cost:            $1.23\n\nProvider limits\n'

test('/fcc-usage opens the panel and draws the fcc report for this session', async ($, on) => {
  const urls: string[] = []
  on('command.register', () => ({ value: { command: 'fcc-usage' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('session.id', () => ({ value: 'sess-123' }))
  on('env.get', () => ({ value: 'http://127.0.0.1:9999/' }))
  on('clock.now', () => ({ value: 1_700_000_000_000 }))
  on('clock.every', () => ({ value: undefined }))
  on('http.fetch', (_, e) => {
    urls.push(e.url)
    return { value: { status: 200, ok: true, headers: {}, text: REPORT } }
  })

  await $.session.start({ cwd: '/', surface: 'terminal', isInteractive: true })
  const ran = await $.command.run({ command: 'fcc-usage', args: 'week' })
  expect(ran.text).toBe('fcc Usage panel opened.')
  expect(urls[0]).toBe('http://127.0.0.1:9999/api/usage/text?period=week&session=sess-123')

  for (const surface of ['terminal', 'desktop', 'vscode', 'mobile'] as const) {
    const ui = await $.ui.mount({ ...PANE, surface })
    expect((await ui.find({ type: 'Text', text: /Total cost:\s+\$1\.23/ }))?.text).toContain('$1.23')
    await ui.unmount()
  }
})

test('the panel says so when fcc is down', async ($, on) => {
  on('command.register', () => ({ value: { command: 'fcc-usage' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('session.id', () => ({ value: 'sess-123' }))
  on('env.get', () => ({ value: undefined }))
  on('clock.every', () => ({ value: undefined }))
  on('http.fetch', () => {
    throw new Error('connection refused')
  })

  await $.session.start({ cwd: '/', surface: 'terminal', isInteractive: true })
  await $.command.run({ command: 'fcc-usage', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await ui.find({ type: 'Text', text: /not reachable/ }))?.text).toContain('http://127.0.0.1:8082')
  await ui.unmount()
})
