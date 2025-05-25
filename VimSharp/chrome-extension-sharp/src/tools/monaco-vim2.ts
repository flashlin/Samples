import * as monaco from 'monaco-editor'

export type VimMode = 'normal' | 'insert'

export interface VimMode2 {
  getMode(): VimMode
  dispose(): void
}

export function initVimMode2(
  editor: monaco.editor.IStandaloneCodeEditor,
  statusBar: HTMLElement
): VimMode2 {
  let mode: VimMode = 'normal'
  updateStatus()

  function updateStatus() {
    statusBar.textContent = `-- ${mode.toUpperCase()} --`
    if (mode === 'normal') {
      editor.updateOptions({ cursorStyle: 'block' })
    } else {
      editor.updateOptions({ cursorStyle: 'line' })
    }
  }

  function handleKeydown(e: monaco.IKeyboardEvent) {
    if (mode === 'normal') {
      if (e.browserEvent.key === 'i') {
        mode = 'insert'
        updateStatus()
        e.preventDefault()
        editor.updateOptions({ readOnly: false })
        return
      }
      // o 新增一行並進入 insert
      if (e.browserEvent.key === 'o') {
        editor.trigger('vim', 'editor.action.insertLineAfter', null)
        mode = 'insert'
        updateStatus()
        e.preventDefault()
        editor.updateOptions({ readOnly: false })
        return
      }
      // hjkl 移動
      if (['h', 'j', 'k', 'l'].includes(e.browserEvent.key)) {
        const command = {
          h: 'cursorLeft',
          j: 'cursorDown',
          k: 'cursorUp',
          l: 'cursorRight'
        }[e.browserEvent.key]!
        editor.trigger('vim', command, null)
        e.preventDefault()
        return
      }
      // w 跳到下一個 word 開始
      if (e.browserEvent.key === 'w') {
        editor.trigger('vim', 'cursorWordStartRight', null)
        e.preventDefault()
        return
      }
      // b 跳到前一個 word
      if (e.browserEvent.key === 'b') {
        editor.trigger('vim', 'cursorWordStartLeft', null)
        e.preventDefault()
        return
      }
      editor.updateOptions({ readOnly: true })
    } else if (mode === 'insert') {
      if (e.browserEvent.key === 'Escape') {
        mode = 'normal'
        updateStatus()
        e.preventDefault()
        editor.updateOptions({ readOnly: true })
        editor.focus()
        return
      }
    }
  }

  editor.updateOptions({ readOnly: true })
  const disposable = editor.onKeyDown(handleKeydown)
  updateStatus()

  return {
    getMode: () => mode,
    dispose: () => {
      disposable.dispose()
    }
  }
} 