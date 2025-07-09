export interface IntellisenseItem {
  title: string
  context: string
  from?: number
  to?: number
}

export interface IntellisenseContext {
  content: string[]
}

export type OnShowIntellisenseFn = (context: IntellisenseContext) => Promise<IntellisenseItem[]>; 