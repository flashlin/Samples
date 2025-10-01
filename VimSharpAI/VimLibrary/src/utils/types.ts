// Common types and interfaces for the library

export interface VimConfig {
  mode?: 'normal' | 'insert' | 'visual'
  readOnly?: boolean
  lineNumbers?: boolean
}

export interface VimEditorOptions {
  initialContent?: string
  config?: VimConfig
}

