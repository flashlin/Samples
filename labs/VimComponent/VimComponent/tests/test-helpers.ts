import { vi } from 'vitest';
import type { VimEditor } from '../src/vim-editor';
import { EditorMode } from '../src/vimEditorTypes';

// P5.js Mock Setup
export function setupP5Mock() {
  vi.mock('p5', () => {
    return {
      default: vi.fn((sketch: any, element: any) => {
        const mockCanvas = {
          elt: document.createElement('canvas'),
          parent: vi.fn(),
        };
        
        const mockP5Instance = {
          setup: vi.fn(),
          draw: vi.fn(),
          createCanvas: vi.fn(() => mockCanvas),
          background: vi.fn(),
          fill: vi.fn(),
          stroke: vi.fn(),
          strokeWeight: vi.fn(),
          line: vi.fn(),
          noFill: vi.fn(),
          noStroke: vi.fn(),
          rect: vi.fn(),
          text: vi.fn(),
          textSize: vi.fn(),
          textAlign: vi.fn(),
          textFont: vi.fn(),
          textWidth: vi.fn(() => 9),
          noLoop: vi.fn(),
          loop: vi.fn(),
          redraw: vi.fn(),
          remove: vi.fn(),
          width: 800,
          height: 600,
          LEFT: 0,
          TOP: 0,
        };
        
        setTimeout(() => {
          if (sketch) {
            sketch(mockP5Instance);
            setTimeout(() => {
              const setupFn = (sketch as any).setup;
              if (setupFn) setupFn.call(mockP5Instance);
            }, 0);
          }
        }, 0);
        
        return mockP5Instance;
      }),
    };
  });
}

// Keyboard Event Helpers
export function dispatchKeyEvent(key: string, target?: any) {
  const editor = target || (globalThis as any).testEditor;
  if (!editor) {
    console.warn('No editor target found for key event:', key);
    return;
  }
  const event = new KeyboardEvent('keydown', { key, bubbles: true });
  editor.dispatchEvent(event);
}

export function pressKey(key: string) {
  dispatchKeyEvent(key);
}

export function pressKeys(...keys: string[]) {
  keys.forEach(key => dispatchKeyEvent(key));
}

// Editor Setup/Teardown
export async function createTestEditor() {
  const editor = document.createElement('vim-editor') as VimEditor;
  document.body.appendChild(editor);
  (globalThis as any).testEditor = editor;
  
  await new Promise(resolve => setTimeout(resolve, 50));
  
  editor.mode = EditorMode.Normal;
  editor.cursorX = 0;
  editor.cursorY = 0;
  (editor as any).hasFocus = true;
  editor.focus();
  
  return editor;
}

export function cleanupTestEditor(editor: any) {
  if (editor && editor.parentNode) {
    editor.parentNode.removeChild(editor);
  }
  (globalThis as any).testEditor = null;
}

// Clipboard Mock
export function setupClipboardMock() {
  const mockReadText = vi.fn();
  const mockWriteText = vi.fn();
  
  Object.defineProperty(navigator, 'clipboard', {
    value: {
      readText: mockReadText,
      writeText: mockWriteText,
    },
    writable: true,
    configurable: true,
  });
  
  return { mockReadText, mockWriteText };
}

