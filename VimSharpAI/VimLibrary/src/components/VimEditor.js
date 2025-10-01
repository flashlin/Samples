/**
 * VimEditor Web Component
 * A custom element that provides a Vim-like editor interface
 */
export class VimEditor extends HTMLElement {
    constructor() {
        super();
        Object.defineProperty(this, "shadow", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "content", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        this.shadow = this.attachShadow({ mode: 'open' });
        this.content = '';
    }
    connectedCallback() {
        this.render();
    }
    /**
     * Initialize the editor with options
     */
    init(options) {
        if (options?.initialContent) {
            this.content = options.initialContent;
        }
        this.render();
    }
    /**
     * Get current content
     */
    getContent() {
        return this.content;
    }
    /**
     * Set content
     */
    setContent(content) {
        this.content = content;
        this.render();
    }
    render() {
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
    `;
    }
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}
// Register the custom element
if (!customElements.get('vim-editor')) {
    customElements.define('vim-editor', VimEditor);
}
// Export for external use
export default VimEditor;
