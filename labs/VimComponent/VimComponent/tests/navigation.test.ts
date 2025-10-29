import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setupP5Mock, createTestEditor, cleanupTestEditor, pressKey, pressKeys } from './test-helpers';

setupP5Mock();

import '../src/vim-editor';

describe('VimEditor - Navigation', () => {
  let editor: any;

  beforeEach(async () => {
    editor = await createTestEditor();
  });

  afterEach(() => {
    cleanupTestEditor(editor);
  });

  describe('$ key in normal mode', () => {
    it('should move cursor to end of line when content is "abc"', () => {
      editor.setContent(['abc']);
      
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
      expect(status.cursorX).toBe(0);
      expect(status.cursorY).toBe(0);
      
      pressKey('$');
      
      const newStatus = editor.getStatus();
      expect(newStatus.cursorX).toBe(2);
      expect(newStatus.cursorY).toBe(0);
      
      editor.updateBuffer();
      
      const buffer = editor.getBuffer();
      expect(buffer[0][0].char).toBe('a');
      expect(buffer[0][1].char).toBe('b');
      expect(buffer[0][2].char).toBe('c');
      
      expect(buffer[0][2].background).toEqual([255, 255, 255]);
      expect(buffer[0][2].foreground).toEqual([0, 0, 0]);
    });

    it('should handle empty line', () => {
      editor.setContent(['']);
      
      pressKey('$');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(0);
    });

    it('should move to last character of long line', () => {
      editor.setContent(['hello world']);
      
      pressKey('$');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(10);
    });
  });

  describe('w key in normal mode', () => {
    it('should move cursor to next English word', () => {
      editor.setContent(['hello world']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      pressKey('w');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(6);
      expect(status.cursorY).toBe(0);
    });

    it('should move cursor to next Chinese word group', () => {
      editor.setContent(['你好世界']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      pressKey('w');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(3);
      expect(status.cursorY).toBe(0);
    });

    it('should move cursor between English and Chinese', () => {
      editor.setContent(['Hello 你好 World']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      pressKey('w');
      
      let status = editor.getStatus();
      expect(status.cursorX).toBe(6);
      
      pressKey('w');
      
      status = editor.getStatus();
      expect(status.cursorX).toBe(9);
      
      pressKey('w');
      
      status = editor.getStatus();
      expect(status.cursorX).toBe(13);
    });

    it('should jump from punctuation to next word', () => {
      editor.setContent(['Hello World中文!']);
      editor.cursorX = 14;
      editor.cursorY = 0;
      
      pressKey('w');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(14);
      expect(status.cursorY).toBe(0);
    });

    it('should move to next line when at end of line', () => {
      editor.setContent(['hello', 'world']);
      editor.cursorX = 4;
      editor.cursorY = 0;
      
      pressKey('w');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(0);
      expect(status.cursorY).toBe(1);
    });
  });

  describe('W key in normal mode', () => {
    it('should move cursor to next space-separated WORD', () => {
      editor.setContent(['hello,world test']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      pressKey('W');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(12);
      expect(status.cursorY).toBe(0);
    });

    it('should treat punctuation as part of WORD', () => {
      editor.setContent(['hello,world! test']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      pressKey('W');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(13);
    });

    it('should move across Chinese and English together', () => {
      editor.setContent(['hello中文world test']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      pressKey('W');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(13);
    });
  });

  describe('B key in normal mode', () => {
    it('should move cursor to previous space-separated WORD', () => {
      editor.setContent(['hello,world test']);
      editor.cursorX = 12;
      editor.cursorY = 0;
      
      pressKey('B');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(0);
      expect(status.cursorY).toBe(0);
    });

    it('should treat punctuation as part of WORD', () => {
      editor.setContent(['hello,world! test']);
      editor.cursorX = 13;
      editor.cursorY = 0;
      
      pressKey('B');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(0);
    });

    it('should move across Chinese and English together', () => {
      editor.setContent(['hello中文world test']);
      editor.cursorX = 13;
      editor.cursorY = 0;
      
      pressKey('B');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(0);
    });
  });

  describe('e key in normal mode', () => {
    it('should move to end of current English word', () => {
      editor.setContent(['hello world']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      pressKey('e');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(4);
      expect(status.cursorY).toBe(0);
    });

    it('should move to end of current Chinese word', () => {
      editor.setContent(['你好世界']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      pressKey('e');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(3);
    });

    it('should stay at same position when cursor is on space', () => {
      editor.setContent(['hello world']);
      editor.cursorX = 5;
      editor.cursorY = 0;
      
      pressKey('e');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(5);
    });

    it('should handle cursor in middle of word', () => {
      editor.setContent(['hello world']);
      editor.cursorX = 2;
      editor.cursorY = 0;
      
      pressKey('e');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(4);
    });

    it('should handle cursor at end of word', () => {
      editor.setContent(['hello world']);
      editor.cursorX = 4;
      editor.cursorY = 0;
      
      pressKey('e');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(10);
    });

    it('should handle mixed English and Chinese', () => {
      editor.setContent(['hello中文world']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      pressKey('e');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(4);
    });

    it('should handle number prefix like 2e', () => {
      editor.setContent(['one two three four']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      pressKey('2');
      
      pressKey('e');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(6);
    });
  });

  describe('^ key in normal mode', () => {
    it('should move cursor to first non-space character', () => {
      editor.setContent(['  hello world']);
      editor.cursorX = 8;
      editor.cursorY = 0;
      
      pressKey('^');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(2);
      expect(status.cursorY).toBe(0);
    });

    it('should move to position 0 if no leading spaces', () => {
      editor.setContent(['hello world']);
      editor.cursorX = 8;
      editor.cursorY = 0;
      
      pressKey('^');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(0);
    });

    it('should handle line with only spaces', () => {
      editor.setContent(['     ']);
      editor.cursorX = 3;
      editor.cursorY = 0;
      
      pressKey('^');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(0);
    });

    it('should handle tabs and spaces', () => {
      editor.setContent(['\t  hello']);
      editor.cursorX = 5;
      editor.cursorY = 0;
      
      pressKey('^');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(3);
    });
  });

  describe('number prefix in normal mode', () => {
    it('should move down 5 lines with 5j', () => {
      editor.setContent(['line1', 'line2', 'line3', 'line4', 'line5', 'line6', 'line7']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('5');
      
      pressKey('j');
      
      const status = editor.getStatus();
      expect(status.cursorY).toBe(5);
    });

    it('should move up 3 lines with 3k', () => {
      editor.setContent(['line1', 'line2', 'line3', 'line4', 'line5', 'line6', 'line7']);
      editor.cursorX = 0;
      editor.cursorY = 6;
      editor.mode = 'normal';
      
      pressKey('3');
      
      pressKey('k');
      
      const status = editor.getStatus();
      expect(status.cursorY).toBe(3);
    });

    it('should support multi-digit numbers like 10j', () => {
      const lines = Array.from({ length: 20 }, (_, i) => `line${i + 1}`);
      editor.setContent(lines);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('1');
      
      pressKey('0');
      
      pressKey('j');
      
      const status = editor.getStatus();
      expect(status.cursorY).toBe(10);
    });

    it('should work with other movement keys like 5w', () => {
      editor.setContent(['one two three four five six seven']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('5');
      
      pressKey('w');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(24);
    });
  });

  describe('viewport scrolling', () => {
    it('should scroll down when cursor moves beyond bottom', () => {
      const lines = Array.from({ length: 50 }, (_, i) => `line${i + 1}`);
      editor.setContent(lines);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      for (let i = 0; i < 30; i++) {
        pressKey('j');
      }
      
      const status = editor.getStatus();
      expect(status.cursorY).toBe(30);
      
      const scroll = editor.getScrollOffset();
      expect(scroll.y).toBeGreaterThan(0);
    });

    it('should scroll up when cursor moves beyond top', () => {
      const lines = Array.from({ length: 50 }, (_, i) => `line${i + 1}`);
      editor.setContent(lines);
      editor.cursorX = 0;
      editor.cursorY = 30;
      editor.mode = 'normal';
      
      for (let i = 0; i < 30; i++) {
        pressKey('j');
      }
      
      for (let i = 0; i < 60; i++) {
        pressKey('k');
      }
      
      const status = editor.getStatus();
      expect(status.cursorY).toBe(0);
      
      const scroll = editor.getScrollOffset();
      expect(scroll.y).toBe(0);
    });

    it('should scroll right when cursor moves beyond right edge', () => {
      const longLine = 'a'.repeat(100);
      editor.setContent([longLine]);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      for (let i = 0; i < 90; i++) {
        pressKey('l');
      }
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(90);
      
      const scroll = editor.getScrollOffset();
      expect(scroll.x).toBeGreaterThan(0);
    });

    it('should scroll left when cursor moves beyond left edge', () => {
      const longLine = 'a'.repeat(100);
      editor.setContent([longLine]);
      editor.cursorX = 90;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      for (let i = 0; i < 90; i++) {
        pressKey('h');
      }
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(0);
      
      const scroll = editor.getScrollOffset();
      expect(scroll.x).toBe(0);
    });

    it('should scroll right when pressing $ to jump to line end', () => {
      const longLine = 'a'.repeat(100);
      editor.setContent([longLine]);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('$');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(99);
      
      const scroll = editor.getScrollOffset();
      expect(scroll.x).toBeGreaterThan(0);
    });
  });

  describe('{ command (previous paragraph)', () => {
    it('should move to previous paragraph', async () => {
      editor.setContent(['Line 1', 'Line 2', '', 'Line 4', 'Line 5']);
      await editor.updateComplete;
      editor.cursorY = 4;
      editor.cursorX = 0;
      editor.mode = 'normal';
      
      await pressKey('{');
      
      expect(editor.cursorY).toBe(3);
      expect(editor.cursorX).toBe(0);
    });

    it('should handle multiple paragraphs', async () => {
      editor.setContent(['P1 Line 1', 'P1 Line 2', '', 'P2 Line 1', '', 'P3 Line 1']);
      await editor.updateComplete;
      editor.cursorY = 5;
      editor.mode = 'normal';
      
      await pressKey('{');
      expect(editor.cursorY).toBe(3);
      
      await pressKey('{');
      expect(editor.cursorY).toBe(0);
    });

    it('should handle consecutive empty lines', async () => {
      editor.setContent(['Line 1', '', '', 'Line 4']);
      await editor.updateComplete;
      editor.cursorY = 3;
      editor.mode = 'normal';
      
      await pressKey('{');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(0);
    });

    it('should stay at beginning when already at first paragraph', async () => {
      editor.setContent(['Line 1', 'Line 2', '', 'Line 4']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      await pressKey('{');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(0);
    });

    it('should handle single paragraph', async () => {
      editor.setContent(['Line 1', 'Line 2', 'Line 3']);
      await editor.updateComplete;
      editor.cursorY = 2;
      editor.mode = 'normal';
      
      await pressKey('{');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(0);
    });
  });

  describe('} command (next paragraph)', () => {
    it('should move to next paragraph', async () => {
      editor.setContent(['Line 1', 'Line 2', '', 'Line 4', 'Line 5']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 0;
      editor.mode = 'normal';
      
      await pressKey('}');
      
      expect(editor.cursorY).toBe(3);
      expect(editor.cursorX).toBe(0);
    });

    it('should handle multiple paragraphs forward', async () => {
      editor.setContent(['P1 Line 1', '', 'P2 Line 1', '', 'P3 Line 1']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      await pressKey('}');
      expect(editor.cursorY).toBe(2);
      
      await pressKey('}');
      expect(editor.cursorY).toBe(4);
    });

    it('should handle consecutive empty lines', async () => {
      editor.setContent(['Line 1', '', '', 'Line 4', 'Line 5']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      await pressKey('}');
      
      expect(editor.cursorY).toBe(3);
      expect(editor.cursorX).toBe(0);
    });

    it('should stay at end when already at last paragraph', async () => {
      editor.setContent(['Line 1', '', 'Line 3', 'Line 4']);
      await editor.updateComplete;
      editor.cursorY = 3;
      editor.mode = 'normal';
      
      await pressKey('}');
      
      expect(editor.cursorY).toBe(3);
    });

    it('should handle single paragraph', async () => {
      editor.setContent(['Line 1', 'Line 2', 'Line 3']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      await pressKey('}');
      
      expect(editor.cursorY).toBe(2);
    });
  });

});

