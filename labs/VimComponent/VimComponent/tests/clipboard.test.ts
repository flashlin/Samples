import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { VimEditor } from '../src/vim-editor';
import { EditorMode } from '../src/vimEditorTypes';
import { createTestEditor, cleanupTestEditor, pressKey, pressKeys, setupP5Mock } from './test-helpers';

setupP5Mock();

let mockClipboardText = '';
let mockReadText: any;
let mockWriteText: any;

mockReadText = vi.fn().mockImplementation(() => 
  Promise.resolve(mockClipboardText)
);

mockWriteText = vi.fn().mockImplementation((text: string) => {
  mockClipboardText = text;
  return Promise.resolve();
});

Object.defineProperty(navigator, 'clipboard', {
  value: {
    readText: mockReadText,
    writeText: mockWriteText,
  },
  writable: true,
  configurable: true,
});

import '../src/vim-editor';

describe('VimEditor - Clipboard Operations', () => {
  let editor: VimEditor;

  beforeEach(async () => {
    mockClipboardText = '';
    editor = await createTestEditor();
  });

  afterEach(() => {
    cleanupTestEditor(editor);
    mockClipboardText = '';
  });

  describe('P command (paste before cursor)', () => {
    it('should paste single line text before cursor', async () => {
      editor.content = ['hello world'];
      editor.cursorY = 0;
      editor.cursorX = 6;
      editor.mode = EditorMode.Normal;
      
      mockClipboardText = 'TEST';
      await pressKey('P');
      await new Promise(resolve => setTimeout(resolve, 10));
      
      expect(editor.content[0]).toBe('hello TESTworld');
      expect(editor.cursorX).toBe(9);
    });
    
    it('should paste at beginning of line', async () => {
      editor.content = ['world'];
      editor.cursorY = 0;
      editor.cursorX = 0;
      
      mockClipboardText = 'hello ';
      await pressKey('P');
      await new Promise(resolve => setTimeout(resolve, 10));
      
      expect(editor.content[0]).toBe('hello world');
      expect(editor.cursorX).toBe(5);
    });
    
    it('should paste multi-line text before cursor', async () => {
      editor.content = ['line 1'];
      editor.cursorY = 0;
      editor.cursorX = 5;
      
      mockClipboardText = 'A\nB\nC';
      await pressKey('P');
      await new Promise(resolve => setTimeout(resolve, 10));
      
      expect(editor.content).toEqual(['line A', 'B', 'C1']);
      expect(editor.cursorY).toBe(2);
      expect(editor.cursorX).toBe(0);
    });
    
    it('should support undo after P', async () => {
      editor.content = ['original'];
      editor.cursorY = 0;
      editor.cursorX = 4;
      
      mockClipboardText = 'NEW';
      await pressKey('P');
      await new Promise(resolve => setTimeout(resolve, 10));
      
      expect(editor.content[0]).toBe('origNEWinal');
      
      await pressKey('u');
      expect(editor.content[0]).toBe('original');
    });
    
    it('should handle Chinese characters', async () => {
      editor.content = ['你好世界'];
      editor.cursorY = 0;
      editor.cursorX = 2;
      
      mockClipboardText = '測試';
      await pressKey('P');
      await new Promise(resolve => setTimeout(resolve, 10));
      
      expect(editor.content[0]).toBe('你好測試世界');
      expect(editor.cursorX).toBe(3);
    });
    
    it('should paste in empty line', async () => {
      editor.content = [''];
      editor.cursorY = 0;
      editor.cursorX = 0;
      
      mockClipboardText = 'content';
      await pressKey('P');
      await new Promise(resolve => setTimeout(resolve, 10));
      
      expect(editor.content[0]).toBe('content');
      expect(editor.cursorX).toBe(6);
    });
  });

  describe('yy command (yank/copy line)', () => {
    it('should copy current line to clipboard with line-wise marker', async () => {
      editor.content = ['line1', 'line2', 'line3'];
      await editor.updateComplete;
      editor.cursorY = 1;
      editor.cursorX = 2;
      editor.mode = EditorMode.Normal;
      
      await pressKeys('y', 'y');
      
      expect(mockClipboardText).toBe('line2\n');
      expect(editor.content).toEqual(['line1', 'line2', 'line3']);
      expect(editor.cursorY).toBe(1);
    });

    it('should allow pasting yanked line with p', async () => {
      editor.content = ['line1', 'line2', 'line3'];
      await editor.updateComplete;
      editor.cursorY = 1;
      editor.cursorX = 0;
      editor.mode = EditorMode.Normal;
      
      await pressKeys('y', 'y');
      
      await pressKey('k');
      expect(editor.cursorY).toBe(0);
      
      await pressKey('p');
      await new Promise(resolve => setTimeout(resolve, 10));
      
      expect(editor.content).toEqual(['line1', 'line2', 'line2', 'line3']);
      expect(editor.cursorY).toBe(1);
    });

    it('should work with empty line', async () => {
      editor.content = ['line1', '', 'line3'];
      await editor.updateComplete;
      editor.cursorY = 1;
      editor.cursorX = 0;
      editor.mode = EditorMode.Normal;
      
      await pressKeys('y', 'y');
      
      expect(mockClipboardText).toBe('\n');
    });
  });

  describe('dd command (delete and copy line)', () => {
    it('should copy line to clipboard before deleting', async () => {
      editor.content = ['line1', 'line2', 'line3'];
      await editor.updateComplete;
      editor.cursorY = 1;
      editor.cursorX = 2;
      editor.mode = EditorMode.Normal;
      
      await pressKeys('d', 'd');
      await new Promise(resolve => setTimeout(resolve, 10));
      
      expect(mockClipboardText).toBe('line2\n');
      expect(editor.content).toEqual(['line1', 'line3']);
      expect(editor.cursorY).toBe(1);
    });

    it('should allow pasting deleted line with p', async () => {
      editor.content = ['line1', 'line2', 'line3'];
      await editor.updateComplete;
      editor.cursorY = 1;
      editor.cursorX = 0;
      editor.mode = EditorMode.Normal;
      
      await pressKeys('d', 'd');
      await new Promise(resolve => setTimeout(resolve, 10));
      
      expect(editor.content).toEqual(['line1', 'line3']);
      expect(editor.cursorY).toBe(1);
      
      await pressKey('k');
      expect(editor.cursorY).toBe(0);
      
      await pressKey('p');
      await new Promise(resolve => setTimeout(resolve, 10));
      
      expect(editor.content).toEqual(['line1', 'line2', 'line3']);
      expect(editor.cursorY).toBe(1);
    });
  });
});

