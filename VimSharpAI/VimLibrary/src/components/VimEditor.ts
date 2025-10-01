import { VimEditorOptions } from '../utils/types'

/**
 * VimEditor Web Component
 * A custom element that provides a Vim-like editor interface
 */
export class VimEditor extends HTMLElement {
  private shadow: ShadowRoot
  private content: string

  constructor() {
    super()
    this.shadow = this.attachShadow({ mode: 'open' })
    this.content = ''
  }

  connectedCallback() {
    this.render()
  }

  /**
   * Initialize the editor with options
   */
  public init(options?: VimEditorOptions) {
    if (options?.initialContent) {
      this.content = options.initialContent
    }
    this.render()
  }

  /**
   * Get current content
   */
  public getContent(): string {
    return this.content
  }

  /**
   * Set content
   */
  public setContent(content: string) {
    this.content = content
    this.render()
  }

  private render() {
    this.shadow.innerHTML = `
      <style>
        :host {
          display: block;
          font-family: monospace;
          border: 1px solid #ccc;
          padding: 10px;
        }
        .editor {
          min-height: 200px;
          background: #1e1e1e;
          color: #d4d4d4;
          padding: 10px;
        }
      </style>
      <div class="editor">
        <pre>${this.escapeHtml(this.content)}</pre>
      </div>
    `
  }

  private escapeHtml(text: string): string {
    const div = document.createElement('div')
    div.textContent = text
    return div.innerHTML
  }
}

// Register the custom element
if (!customElements.get('vim-editor')) {
  customElements.define('vim-editor', VimEditor)
}

// Export for external use
export default VimEditor

