import { describe, it, expect, beforeEach, afterEach, beforeAll, afterAll } from 'vitest';
import { VimEditor } from '../src/vim-editor';
import { EditorMode } from '../src/vimEditorTypes';
import { createTestEditor, cleanupTestEditor, pressKey, pressKeys } from './test-helpers';

describe.skip.sequential('VimEditor - Clipboard Operations', () => {
  let editor: VimEditor;

  beforeAll(async () => {
    await new Promise(resolve => setTimeout(resolve, 100));
    await navigator.clipboard.writeText('');
    await new Promise(resolve => setTimeout(resolve, 50));
  });

  beforeEach(async () => {
    editor = await createTestEditor();
    await navigator.clipboard.writeText('');
    await new Promise(resolve => setTimeout(resolve, 50));
  });

  afterEach(async () => {
    cleanupTestEditor(editor);
    await navigator.clipboard.writeText('');
    await new Promise(resolve => setTimeout(resolve, 50));
  });

  afterAll(async () => {
    await navigator.clipboard.writeText('');
    await new Promise(resolve => setTimeout(resolve, 50));
  });

  describe('P command (paste before cursor)', () => {
    it.sequential('should paste single line text before cursor', async () => {
      await navigator.clipboard.writeText('');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      editor.content = ['hello world'];
      editor.cursorY = 0;
      editor.cursorX = 6;
      editor.mode = EditorMode.Normal;
      
      await navigator.clipboard.writeText('TEST');
      await new Promise(resolve => setTimeout(resolve, 50));
      await pressKey('P');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content[0]).toBe('hello TESTworld');
      expect(editor.cursorX).toBe(9);
    });
    
    it.sequential('should paste at beginning of line', async () => {
      await navigator.clipboard.writeText('');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      editor.content = ['world'];
      editor.cursorY = 0;
      editor.cursorX = 0;
      
      await navigator.clipboard.writeText('hello ');
      await new Promise(resolve => setTimeout(resolve, 50));
      await pressKey('P');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content[0]).toBe('hello world');
      expect(editor.cursorX).toBe(5);
    });
    
    it.sequential('should paste multi-line text before cursor', async () => {
      await navigator.clipboard.writeText('');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      editor.content = ['line 1'];
      editor.cursorY = 0;
      editor.cursorX = 5;
      
      await navigator.clipboard.writeText('A\nB\nC');
      await new Promise(resolve => setTimeout(resolve, 50));
      await pressKey('P');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content).toEqual(['line A', 'B', 'C1']);
      expect(editor.cursorY).toBe(2);
      expect(editor.cursorX).toBe(0);
    });
    
    it.sequential('should support undo after P', async () => {
      await navigator.clipboard.writeText('');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      editor.content = ['original'];
      editor.cursorY = 0;
      editor.cursorX = 4;
      
      await navigator.clipboard.writeText('NEW');
      await new Promise(resolve => setTimeout(resolve, 50));
      await pressKey('P');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content[0]).toBe('origNEWinal');
      
      await pressKey('u');
      expect(editor.content[0]).toBe('original');
    });
    
    it.sequential('should handle Chinese characters', async () => {
      await navigator.clipboard.writeText('');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      editor.content = ['你好世界'];
      editor.cursorY = 0;
      editor.cursorX = 2;
      
      await navigator.clipboard.writeText('測試');
      await new Promise(resolve => setTimeout(resolve, 50));
      await pressKey('P');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content[0]).toBe('你好測試世界');
      expect(editor.cursorX).toBe(3);
    });
    
    it.sequential('should paste in empty line', async () => {
      await navigator.clipboard.writeText('');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      editor.content = [''];
      editor.cursorY = 0;
      editor.cursorX = 0;
      
      await navigator.clipboard.writeText('content');
      await new Promise(resolve => setTimeout(resolve, 50));
      await pressKey('P');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content[0]).toBe('content');
      expect(editor.cursorX).toBe(6);
    });
  });

  describe('yy command (yank/copy line)', () => {
    it.sequential('should copy current line to clipboard with line-wise marker', async () => {
      await navigator.clipboard.writeText('');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      editor.content = ['line1', 'line2', 'line3'];
      await editor.updateComplete;
      editor.cursorY = 1;
      editor.cursorX = 2;
      editor.mode = EditorMode.Normal;
      
      await pressKeys('y', 'y');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      const clipboardText = await navigator.clipboard.readText();
      expect(clipboardText).toBe('line2\n');
      
      expect(editor.content).toEqual(['line1', 'line2', 'line3']);
      expect(editor.cursorY).toBe(1);
    });

    it.sequential('should allow pasting yanked line with p', async () => {
      await navigator.clipboard.writeText('');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      editor.content = ['line1', 'line2', 'line3'];
      await editor.updateComplete;
      editor.cursorY = 1;
      editor.cursorX = 0;
      editor.mode = EditorMode.Normal;
      
      await pressKeys('y', 'y');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      await pressKey('k');
      expect(editor.cursorY).toBe(0);
      
      await pressKey('p');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content).toEqual(['line1', 'line2', 'line2', 'line3']);
      expect(editor.cursorY).toBe(1);
    });

    it.sequential('should work with empty line', async () => {
      await navigator.clipboard.writeText('');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      editor.content = ['line1', '', 'line3'];
      await editor.updateComplete;
      editor.cursorY = 1;
      editor.cursorX = 0;
      editor.mode = EditorMode.Normal;
      
      await pressKeys('y', 'y');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      const clipboardText = await navigator.clipboard.readText();
      expect(clipboardText).toBe('\n');
    });
  });

  describe('dd command (delete and copy line)', () => {
    it.sequential('should copy line to clipboard before deleting', async () => {
      await navigator.clipboard.writeText('');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      editor.content = ['line1', 'line2', 'line3'];
      await editor.updateComplete;
      editor.cursorY = 1;
      editor.cursorX = 2;
      editor.mode = EditorMode.Normal;
      
      await pressKeys('d', 'd');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      const clipboardText = await navigator.clipboard.readText();
      expect(clipboardText).toBe('line2\n');
      
      expect(editor.content).toEqual(['line1', 'line3']);
      expect(editor.cursorY).toBe(1);
    });

    it.sequential('should allow pasting deleted line with p', async () => {
      await navigator.clipboard.writeText('');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      editor.content = ['line1', 'line2', 'line3'];
      await editor.updateComplete;
      editor.cursorY = 1;
      editor.cursorX = 0;
      editor.mode = EditorMode.Normal;
      
      await pressKeys('d', 'd');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content).toEqual(['line1', 'line3']);
      expect(editor.cursorY).toBe(1);
      
      await pressKey('k');
      expect(editor.cursorY).toBe(0);
      
      await pressKey('p');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content).toEqual(['line1', 'line2', 'line3']);
      expect(editor.cursorY).toBe(1);
    });
  });
});

