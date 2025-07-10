export interface IntellisenseItem {
  title: string
  getFromPosition?: (fromPosition: number) => number  
  getContext(): string
}

export interface IntellisenseContext {
  content: string[]
}

export type OnShowIntellisenseFn = (context: IntellisenseContext) => Promise<IntellisenseItem[]>; 