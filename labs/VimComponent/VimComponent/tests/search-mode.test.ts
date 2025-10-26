import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setupP5Mock, createTestEditor, cleanupTestEditor, pressKey, pressKeys } from './test-helpers';

setupP5Mock();

import '../src/vim-editor';

describe('VimEditor - Search Mode', () => {
  let editor: any;

  beforeEach(async () => {
    editor = await createTestEditor();
  });

  afterEach(() => {
    cleanupTestEditor(editor);
  });

  describe('search mode (*)', () => {
    it('should enter search mode with * from visual selection', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('fast-search');
      expect(status.searchKeyword).toBe('hello');
      expect(status.searchMatchCount).toBe(3);
    });

    it('should highlight all search matches', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(0);
    });

    it('should jump to next match with n', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      
      pressKey('n');
      expect(editor.cursorY).toBe(1);
      expect(editor.cursorX).toBe(5);
      
      pressKey('n');
      expect(editor.cursorY).toBe(2);
      expect(editor.cursorX).toBe(0);
    });

    it('should jump to previous match with N', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      pressKeys('n', 'n');
      
      pressKey('N');
      expect(editor.cursorY).toBe(1);
      expect(editor.cursorX).toBe(5);
    });

    it('should wrap around when jumping to next/previous', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      pressKeys('n', 'n', 'n');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(0);
    });

    it('should clear current search mark with b', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      
      expect(editor.getStatus().searchMatchCount).toBe(3);
      
      pressKey('b');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('fast-search');
      expect(status.searchKeyword).toBe('hello');
      expect(status.searchMatchCount).toBe(2);
    });

    it('should restore search marks with u', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      expect(editor.getStatus().searchMatchCount).toBe(3);
      
      pressKey('b');
      expect(editor.getStatus().searchMatchCount).toBe(2);
      
      pressKey('u');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('fast-search');
      expect(status.searchKeyword).toBe('hello');
      expect(status.searchMatchCount).toBe(3);
    });

    it('should clear multiple marks one by one with b', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      expect(editor.getStatus().searchMatchCount).toBe(3);
      
      pressKey('b');
      expect(editor.getStatus().searchMatchCount).toBe(2);
      expect(editor.getStatus().mode).toBe('fast-search');
      
      pressKey('b');
      expect(editor.getStatus().searchMatchCount).toBe(1);
      expect(editor.getStatus().mode).toBe('fast-search');
      
      pressKey('b');
      expect(editor.getStatus().searchMatchCount).toBe(undefined);
      expect(editor.getStatus().mode).toBe('normal');
    });

    it('should remove current mark and jump to next', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(0);
      
      pressKey('b');
      
      expect(editor.cursorY).toBe(1);
      expect(editor.cursorX).toBe(5);
    });

    it('should restore marks in correct order', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      pressKeys('b', 'b');
      expect(editor.getStatus().searchMatchCount).toBe(1);
      
      pressKey('u');
      expect(editor.getStatus().searchMatchCount).toBe(2);
      
      pressKey('u');
      expect(editor.getStatus().searchMatchCount).toBe(3);
    });

    it('should enter multi-insert mode with i', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      pressKey('i');
      
      expect(editor.getStatus().mode).toBe('multi-insert');
    });

    it('should exit search mode with Escape', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      pressKey('Escape');
      
      expect(editor.getStatus().mode).toBe('normal');
    });
  });
  describe('search mode cursor movement', () => {
    it('should move cursor left with h in search mode', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      
      expect(editor.cursorX).toBe(0);
      
      pressKeys('$', 'h');
      
      expect(editor.cursorX).toBe(3);
    });

    it('should move cursor right with l in search mode', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      
      expect(editor.cursorX).toBe(0);
      
      pressKey('l');
      
      expect(editor.cursorX).toBe(1);
    });

    it('should move to start of match with 0', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*', 'l', 'l', '0');
      
      expect(editor.cursorX).toBe(0);
    });

    it('should move to end of match with $', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*', '$');
      
      expect(editor.cursorX).toBe(4);
    });

    it('should not move beyond match boundaries', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      
      pressKeys('l', 'l', 'l', 'l', 'l', 'l');
      
      expect(editor.cursorX).toBe(4);
    });

    it('should delete character at cursor with x', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*', 'l', 'x');
      
      expect(editor.content).toEqual([
        'hllo world',
        'test hllo test',
        'hllo again'
      ]);
      expect(editor.getStatus().searchKeyword).toBe('hllo');
    });

    it('should delete all matches with d', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*', 'd');
      
      expect(editor.content).toEqual([
        ' world',
        'test  test',
        ' again'
      ]);
      expect(editor.mode).toBe('normal');
    });
  });
  describe('multi-insert mode', () => {
    it('should enter multi-insert at current cursor position with i', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      expect(editor.cursorX).toBe(0);
      
      pressKeys('l', 'l', 'i');
      
      expect(editor.mode).toBe('multi-insert');
      expect(editor.cursorX).toBe(2);
    });

    it('should enter multi-insert at next position with a', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      expect(editor.cursorX).toBe(0);
      
      pressKeys('l', 'l', 'a');
      
      expect(editor.mode).toBe('multi-insert');
      expect(editor.cursorX).toBe(3);
    });

    it('should insert at cursor position not at end', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      pressKeys('l', 'i', 'X');
      
      expect(editor.content).toEqual([
        'hXello world',
        'test hXello test',
        'hXello again'
      ]);
    });

    it('should insert after cursor with a key', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*');
      pressKeys('l', 'a', 'X');
      
      expect(editor.content).toEqual([
        'heXllo world',
        'test heXllo test',
        'heXllo again'
      ]);
    });

    it('should insert character at all search matches', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*', '$', 'a', 'X');
      
      expect(editor.content).toEqual([
        'helloX world',
        'test helloX test',
        'helloX again'
      ]);
    });

    it('should delete character from all search matches with Backspace', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*', 'i', 'X', 'Backspace');
      
      expect(editor.content).toEqual([
        'hello world',
        'test hello test',
        'hello again'
      ]);
    });

    it('should insert newline at all search matches', async () => {
      editor.setContent(['hello world', 'test hello test']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*', '$', 'a', 'Enter');
      
      expect(editor.content).toEqual([
        'hello',
        ' world',
        'test hello',
        ' test'
      ]);
    });

    it('should exit multi-insert mode with Escape to search mode', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*', 'i');
      expect(editor.getStatus().mode).toBe('multi-insert');
      
      pressKey('Escape');
      expect(editor.getStatus().mode).toBe('fast-search');
      
      pressKey('Escape');
      expect(editor.getStatus().mode).toBe('normal');
    });

    it('should handle multiple character insertions', async () => {
      editor.setContent(['hello world', 'test hello test', 'hello again']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'l', 'l', 'l', 'l', '*', '$', 'a', 'X', 'Y', 'Z');
      
      expect(editor.content).toEqual([
        'helloXYZ world',
        'test helloXYZ test',
        'helloXYZ again'
      ]);
    });
  });
});
