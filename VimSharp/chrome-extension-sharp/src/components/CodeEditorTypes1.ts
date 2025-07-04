export interface IntellisenseItem {
  title: string
  context: string
}

export interface IntellisenseContext {
  content: string[]
}

export type OnShowIntellisenseFn = (context: IntellisenseContext) => Promise<IntellisenseItem[]>; 