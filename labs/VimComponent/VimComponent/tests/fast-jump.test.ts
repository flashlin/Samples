import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setupP5Mock, createTestEditor, cleanupTestEditor, pressKey, pressKeys } from './test-helpers';

setupP5Mock();

import '../src/vim-editor';

describe('VimEditor - Fast Jump', () => {
  let editor: any;

  beforeEach(async () => {
    editor = await createTestEditor();
  });

  afterEach(() => {
    cleanupTestEditor(editor);
  });

  describe('fast-jump mode', () => {
    it('should enter fast-jump mode when pressing f', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      
      pressKey('f');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('fast-jump');
    });

    it('should return to normal mode if no matches found', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      
      pressKey('f');
      
      pressKey('x');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
    });

    it('should jump directly if only one match found', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      pressKey('f');
      
      pressKey('w');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
      expect(status.cursorX).toBe(6);
      expect(status.cursorY).toBe(0);
    });

    it('should enter match mode if multiple matches found', async () => {
      editor.setContent(['hello hello hello']);
      await editor.updateComplete;
      
      pressKey('f');
      
      pressKey('h');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('fast-match');
    });

    it('should jump to correct position after selecting label', async () => {
      editor.setContent(['hello hello hello']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      pressKey('f');
      
      pressKey('h');
      
      pressKey('b');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
      expect(status.cursorX).toBe(6);
      expect(status.cursorY).toBe(0);
    });

    it('should filter matches as user types label', async () => {
      editor.setContent(['aaa bbb ccc ddd eee fff ggg hhh iii jjj kkk lll mmm nnn ooo ppp qqq rrr sss ttt uuu vvv www xxx yyy zzz']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      pressKeys('f', 'a');
      
      let status = editor.getStatus();
      expect(status.mode).toBe('fast-match');
      
      pressKey('b');
      
      status = editor.getStatus();
      expect(status.mode).toBe('normal');
      expect(status.cursorX).toBe(1);
      expect(status.cursorY).toBe(0);
    });

    it('should exit on Escape in fast-jump mode', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      
      pressKey('f');
      
      pressKey('Escape');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
    });

    it('should exit on Escape in match mode', async () => {
      editor.setContent(['hello hello hello']);
      await editor.updateComplete;
      
      pressKey('f');
      
      pressKey('h');
      
      pressKey('Escape');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
    });

    it('should work in visual mode', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      pressKey('v');
      
      expect(editor.getStatus().mode).toBe('visual');
      
      pressKey('f');
      
      expect(editor.getStatus().mode).toBe('fast-jump');
      
      pressKey('w');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('visual');
      expect(status.cursorX).toBe(6);
      expect(status.cursorY).toBe(0);
    });

    it('should work in visual-line mode', async () => {
      editor.setContent(['hello world', 'foo bar baz']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      pressKey('V');
      
      expect(editor.getStatus().mode).toBe('visual-line');
      
      pressKey('f');
      
      expect(editor.getStatus().mode).toBe('fast-jump');
      
      pressKey('b');
      
      expect(editor.getStatus().mode).toBe('fast-match');
      
      pressKey('b');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('visual-line');
      expect(status.cursorX).toBe(8);
      expect(status.cursorY).toBe(1);
    });

    it('should return to visual mode when pressing Escape in fast-jump', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      
      pressKey('v');
      
      pressKey('f');
      
      pressKey('Escape');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('visual');
    });

    it('should return to visual mode when pressing Escape in match mode', async () => {
      editor.setContent(['hello hello hello']);
      await editor.updateComplete;
      
      pressKey('v');
      
      pressKey('f');
      
      pressKey('h');
      
      expect(editor.getStatus().mode).toBe('fast-match');
      
      pressKey('Escape');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('visual');
    });
  });
});
