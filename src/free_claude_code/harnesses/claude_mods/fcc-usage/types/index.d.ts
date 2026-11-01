export type Period = 'day' | 'week'

export type Report = {
  text: string | null
  error: string | null
  period: Period
  loading: boolean
  updatedAt: number | null
}

declare module 'claude-code' {
  interface PluginState {
    'fcc-usage': { report: Report }
  }
}
