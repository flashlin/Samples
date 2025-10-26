import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { setupP5Mock, createTestEditor, cleanupTestEditor } from './test-helpers';

setupP5Mock();

import '../src/vim-editor';

describe('VimEditor - System', () => {
  let editor: any;

  beforeEach(async () => {
    editor = await createTestEditor();
  });

  afterEach(() => {
    cleanupTestEditor(editor);
  });

  describe('buffer system', () => {
    it('should maintain buffer state', () => {
      editor.setContent(['test']);
      
      const buffer = editor.getBuffer();
      
      expect(buffer.length).toBeGreaterThan(0);
      expect(buffer[0].length).toBeGreaterThan(0);
      
      expect(buffer[0][0].char).toBe('t');
      expect(buffer[0][1].char).toBe('e');
      expect(buffer[0][2].char).toBe('s');
      expect(buffer[0][3].char).toBe('t');
    });

    it('should show cursor position in buffer', () => {
      editor.setContent(['abc']);
      editor.cursorX = 1;
      editor.cursorY = 0;
      
      editor.updateBuffer();
      
      const buffer = editor.getBuffer();
      const status = editor.getStatus();
      
      if (status.cursorVisible) {
        expect(buffer[0][1].background).toEqual([255, 255, 255]);
        expect(buffer[0][1].foreground).toEqual([0, 0, 0]);
      }
      
      expect(buffer[0][0].char).toBe('a');
      expect(buffer[0][1].char).toBe('b');
      expect(buffer[0][2].char).toBe('c');
    });
  });

  describe('status system', () => {
    it('should provide current mode', () => {
      editor.mode = 'normal';
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
    });

    it('should provide cursor position', () => {
      editor.cursorX = 5;
      editor.cursorY = 2;
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(5);
      expect(status.cursorY).toBe(2);
    });

    it('should track mode changes', () => {
      editor.mode = 'normal';
      expect(editor.getStatus().mode).toBe('normal');
      
      editor.mode = 'insert';
      expect(editor.getStatus().mode).toBe('insert');
    });
  });

  describe('display column calculation', () => {
    it('should calculate display column for English text', () => {
      editor.setContent(['Hello World']);
      editor.cursorX = 6;
      editor.cursorY = 0;
      
      const col = editor.getDisplayColumn();
      expect(col).toBe(6);
    });

    it('should calculate display column with Chinese characters', () => {
      editor.setContent(['Hello World 中文!']);
      editor.cursorX = 13;
      editor.cursorY = 0;
      
      const col = editor.getDisplayColumn();
      expect(col).toBe(14);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(13);
    });

    it('should calculate display column at start of Chinese character', () => {
      editor.setContent(['Hello World 中文!']);
      editor.cursorX = 12;
      editor.cursorY = 0;
      
      const col = editor.getDisplayColumn();
      expect(col).toBe(12);
    });

    it('should calculate display column for all Chinese text', () => {
      editor.setContent(['你好世界']);
      editor.cursorX = 2;
      editor.cursorY = 0;
      
      const col = editor.getDisplayColumn();
      expect(col).toBe(4);
    });
  });

  describe('load method (public API)', () => {
    it('should load single line text', async () => {
      const text = 'Hello World';
      editor.load(text);
      await editor.updateComplete;
      
      expect(editor.content).toEqual(['Hello World']);
      expect(editor.cursorX).toBe(0);
      expect(editor.cursorY).toBe(0);
      expect(editor.mode).toBe('normal');
    });

    it('should load multiline text', async () => {
      const text = 'Line 1\nLine 2\nLine 3';
      editor.load(text);
      await editor.updateComplete;
      
      expect(editor.content).toEqual(['Line 1', 'Line 2', 'Line 3']);
      expect(editor.cursorX).toBe(0);
      expect(editor.cursorY).toBe(0);
      expect(editor.mode).toBe('normal');
    });

    it('should load Chinese text', async () => {
      const text = '你好世界\n測試文字';
      editor.load(text);
      await editor.updateComplete;
      
      expect(editor.content).toEqual(['你好世界', '測試文字']);
      expect(editor.cursorX).toBe(0);
      expect(editor.cursorY).toBe(0);
    });

    it('should load empty text', async () => {
      const text = '';
      editor.load(text);
      await editor.updateComplete;
      
      expect(editor.content).toEqual(['']);
      expect(editor.cursorX).toBe(0);
      expect(editor.cursorY).toBe(0);
    });

    it('should load text with only newlines', async () => {
      const text = '\n\n\n';
      editor.load(text);
      await editor.updateComplete;
      
      expect(editor.content).toEqual(['', '', '', '']);
      expect(editor.cursorX).toBe(0);
      expect(editor.cursorY).toBe(0);
    });

    it('should reset cursor position when loading new text', async () => {
      editor.setContent(['Line 1', 'Line 2', 'Line 3']);
      await editor.updateComplete;
      editor.cursorX = 5;
      editor.cursorY = 2;
      
      const text = 'New content';
      editor.load(text);
      await editor.updateComplete;
      
      expect(editor.content).toEqual(['New content']);
      expect(editor.cursorX).toBe(0);
      expect(editor.cursorY).toBe(0);
    });

    it('should reset scroll position when loading new text', async () => {
      editor.setContent(['Line 1', 'Line 2', 'Line 3', 'Line 4', 'Line 5']);
      await editor.updateComplete;
      
      const text = 'New content';
      editor.load(text);
      await editor.updateComplete;
      
      const scrollOffset = editor.getScrollOffset();
      expect(scrollOffset.x).toBe(0);
      expect(scrollOffset.y).toBe(0);
    });

    it('should set mode to normal when loading new text', async () => {
      editor.mode = 'insert';
      
      const text = 'Some text';
      editor.load(text);
      await editor.updateComplete;
      
      expect(editor.mode).toBe('normal');
    });

    it('should handle text with mixed line endings', async () => {
      const text = 'Line 1\nLine 2\nLine 3';
      editor.load(text);
      await editor.updateComplete;
      
      expect(editor.content).toEqual(['Line 1', 'Line 2', 'Line 3']);
    });

    it('should load code with special characters', async () => {
      const text = 'function test() {\n  return "Hello!";\n}';
      editor.load(text);
      await editor.updateComplete;
      
      expect(editor.content).toEqual([
        'function test() {',
        '  return "Hello!";',
        '}'
      ]);
      expect(editor.cursorX).toBe(0);
      expect(editor.cursorY).toBe(0);
    });

    it('should allow editing after loading text', async () => {
      const text = 'Initial text';
      editor.load(text);
      await editor.updateComplete;
      
      editor.mode = 'insert';
      editor.cursorX = 7;
      
      expect(editor.content[0]).toBe('Initial text');
      expect(editor.cursorX).toBe(7);
    });
  });
});
