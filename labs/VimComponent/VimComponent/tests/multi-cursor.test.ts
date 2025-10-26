import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setupP5Mock, createTestEditor, cleanupTestEditor, pressKey, pressKeys, setupClipboardMock } from './test-helpers';

setupP5Mock();

import '../src/vim-editor';

describe('VimEditor - Multi-Cursor Editing', () => {
  let editor: any;

  beforeEach(async () => {
    editor = await createTestEditor();
  });

  afterEach(() => {
    cleanupTestEditor(editor);
  });

  describe('t-insert mode (multi-cursor editing)', () => {
    it('should mark positions with t key in normal mode', async () => {
      editor.setContent(['hello world', 'test code', 'example']);
      editor.cursorX = 6;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('t');
      
      expect(editor.tMarks.length).toBe(1);
      expect(editor.tMarks[0]).toEqual({ x: 6, y: 0 });
    });

    it('should accumulate multiple t-marks', async () => {
      editor.setContent(['hello world', 'test code', 'example']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      editor.cursorX = 6;
      editor.cursorY = 0;
      pressKey('t');
      
      editor.cursorX = 5;
      editor.cursorY = 1;
      pressKey('t');
      
      editor.cursorX = 3;
      editor.cursorY = 2;
      pressKey('t');
      
      expect(editor.tMarks.length).toBe(3);
      expect(editor.tMarks).toEqual([
        { x: 6, y: 0 },
        { x: 5, y: 1 },
        { x: 3, y: 2 }
      ]);
    });

    it('should enter t-insert mode when pressing i with t-marks', async () => {
      editor.setContent(['hello world']);
      editor.cursorX = 6;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('t');
      pressKey('i');
      
      expect(editor.mode).toBe('t-insert');
      expect(editor.cursorX).toBe(6);
      expect(editor.cursorY).toBe(0);
    });

    it('should insert characters at all t-mark positions', async () => {
      editor.setContent(['hello world', 'test code', 'example']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      editor.cursorX = 6;
      editor.cursorY = 0;
      pressKey('t');
      
      editor.cursorX = 5;
      editor.cursorY = 1;
      pressKey('t');
      
      editor.cursorX = 3;
      editor.cursorY = 2;
      pressKey('t');
      
      pressKey('i');
      
      const hiddenInput = editor.shadowRoot?.querySelector('input');
      if (hiddenInput) {
        hiddenInput.value = 'X';
        const inputEvent = new Event('input', { bubbles: true });
        hiddenInput.dispatchEvent(inputEvent);
        await editor.updateComplete;
        
        expect(editor.content[0]).toBe('hello Xworld');
        expect(editor.content[1]).toBe('test Xcode');
        expect(editor.content[2]).toBe('exaXmple');
      }
    });

    it('should keep t-marks position fixed while cursor moves', async () => {
      editor.setContent(['hello world']);
      editor.cursorX = 6;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('t');
      const originalMark = { ...editor.tMarks[0] };
      
      pressKey('i');
      
      const hiddenInput = editor.shadowRoot?.querySelector('input');
      if (hiddenInput) {
        hiddenInput.value = 'X';
        const inputEvent = new Event('input', { bubbles: true });
        hiddenInput.dispatchEvent(inputEvent);
        await editor.updateComplete;
        
        expect(editor.tMarks[0]).toEqual(originalMark);
        expect(editor.cursorX).toBe(7);
      }
    });

    it('should insert multiple characters at all t-mark positions', async () => {
      editor.setContent(['ab cd', 'ef gh']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      editor.cursorX = 3;
      editor.cursorY = 0;
      pressKey('t');
      
      editor.cursorX = 3;
      editor.cursorY = 1;
      pressKey('t');
      
      pressKey('i');
      
      const hiddenInput = editor.shadowRoot?.querySelector('input');
      if (hiddenInput) {
        hiddenInput.value = 'XYZ';
        const inputEvent = new Event('input', { bubbles: true });
        hiddenInput.dispatchEvent(inputEvent);
        await editor.updateComplete;
        
        expect(editor.content[0]).toBe('ab XYZcd');
        expect(editor.content[1]).toBe('ef XYZgh');
        expect(editor.cursorX).toBe(6);
      }
    });

    it('should delete characters at all t-mark positions with backspace', async () => {
      editor.setContent(['hello world', 'test code']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      editor.cursorX = 6;
      editor.cursorY = 0;
      pressKey('t');
      
      editor.cursorX = 5;
      editor.cursorY = 1;
      pressKey('t');
      
      pressKey('i');
      
      const hiddenInput = editor.shadowRoot?.querySelector('input');
      if (hiddenInput) {
        hiddenInput.value = 'X';
        let inputEvent = new Event('input', { bubbles: true });
        hiddenInput.dispatchEvent(inputEvent);
        await editor.updateComplete;
        
        expect(editor.content[0]).toBe('hello Xworld');
        expect(editor.content[1]).toBe('test Xcode');
        
        pressKey('Backspace');
        await editor.updateComplete;
        
        expect(editor.content[0]).toBe('hello world');
        expect(editor.content[1]).toBe('test code');
      }
    });

    it('should clear all t-marks when pressing Escape in normal mode', async () => {
      editor.setContent(['hello world', 'test code']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      editor.cursorX = 6;
      editor.cursorY = 0;
      pressKey('t');
      
      editor.cursorX = 5;
      editor.cursorY = 1;
      pressKey('t');
      
      expect(editor.tMarks.length).toBe(2);
      
      pressKey('Escape');
      
      expect(editor.tMarks.length).toBe(0);
    });

    it('should exit t-insert mode and return to normal with Escape', async () => {
      editor.setContent(['hello world']);
      editor.cursorX = 6;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('t');
      pressKey('i');
      
      expect(editor.mode).toBe('t-insert');
      
      pressKey('Escape');
      
      expect(editor.mode).toBe('normal');
    });

    it('should preserve t-marks after editing and exiting t-insert mode', async () => {
      editor.setContent(['hello world']);
      editor.cursorX = 6;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('t');
      const originalMark = { ...editor.tMarks[0] };
      
      pressKey('i');
      
      const hiddenInput = editor.shadowRoot?.querySelector('input');
      if (hiddenInput) {
        hiddenInput.value = 'X';
        const inputEvent = new Event('input', { bubbles: true });
        hiddenInput.dispatchEvent(inputEvent);
        await editor.updateComplete;
      }
      
      pressKey('Escape');
      
      expect(editor.tMarks.length).toBe(1);
      expect(editor.tMarks[0]).toEqual(originalMark);
    });

    it('should handle t-marks at different positions on same line', async () => {
      editor.setContent(['a b c d e']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      editor.cursorX = 2;
      editor.cursorY = 0;
      pressKey('t');
      
      editor.cursorX = 6;
      editor.cursorY = 0;
      pressKey('t');
      
      pressKey('i');
      
      const hiddenInput = editor.shadowRoot?.querySelector('input');
      if (hiddenInput) {
        hiddenInput.value = 'X';
        const inputEvent = new Event('input', { bubbles: true });
        hiddenInput.dispatchEvent(inputEvent);
        await editor.updateComplete;
        
        expect(editor.content[0]).toBe('a Xb c Xd e');
      }
    });

    it('should not create duplicate t-marks at same position', async () => {
      editor.setContent(['hello world']);
      editor.cursorX = 6;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('t');
      pressKey('t');
      pressKey('t');
      
      expect(editor.tMarks.length).toBe(1);
    });

    it('should mark next position when pressing T in normal mode', async () => {
      editor.setContent(['hello world']);
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('T');
      
      expect(editor.tMarks.length).toBe(1);
      expect(editor.tMarks[0]).toEqual({ x: 6, y: 0 });
    });

    it('should mark next line start when T pressed at end of line', async () => {
      editor.setContent(['hello', 'world']);
      editor.cursorX = 4;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('T');
      
      expect(editor.tMarks.length).toBe(1);
      expect(editor.tMarks[0]).toEqual({ x: 0, y: 1 });
    });

    it('should handle T at the end of last line', async () => {
      editor.setContent(['hello world']);
      editor.cursorX = 10;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('T');
      
      expect(editor.tMarks.length).toBe(1);
      expect(editor.tMarks[0]).toEqual({ x: 10, y: 0 });
    });

    it('should accumulate multiple t-marks with t and T', async () => {
      editor.setContent(['hello world', 'test code']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      editor.cursorX = 0;
      editor.cursorY = 0;
      pressKey('t');
      
      editor.cursorX = 5;
      editor.cursorY = 0;
      pressKey('T');
      
      editor.cursorX = 3;
      editor.cursorY = 1;
      pressKey('t');
      
      expect(editor.tMarks.length).toBe(3);
      expect(editor.tMarks).toEqual([
        { x: 0, y: 0 },
        { x: 6, y: 0 },
        { x: 3, y: 1 }
      ]);
    });
  });
  describe('TVisual mode (multi-cursor visual)', () => {
    it('should enter TVisual mode when pressing v at tMark position', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      // Add some tMarks
      editor.cursorX = 0;
      editor.cursorY = 0;
      pressKey('t');
      
      editor.cursorX = 6;
      editor.cursorY = 0;
      pressKey('t');
      
      // Press v at a tMark position
      editor.cursorX = 0;
      editor.cursorY = 0;
      pressKey('v');
      
      expect(editor.mode).toBe('t-visual');
    });

    it('should enter normal visual mode when pressing v at non-tMark position', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      // Add a tMark
      editor.cursorX = 0;
      editor.cursorY = 0;
      pressKey('t');
      
      // Press v at a non-tMark position
      editor.cursorX = 3;
      editor.cursorY = 0;
      pressKey('v');
      
      expect(editor.mode).toBe('visual');
    });

    it('should yank multiple selections independently with e key in TVisual mode', async () => {
      editor.setContent(['tools app', 'hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      // Add tMarks: 't' in "tools" and 'h' in "hello"
      editor.cursorX = 0;
      editor.cursorY = 0;
      pressKey('t');
      
      editor.cursorX = 0;
      editor.cursorY = 1;
      pressKey('t');
      
      // Start TVisual mode from first tMark
      editor.cursorX = 0;
      editor.cursorY = 0;
      pressKey('v');
      
      expect(editor.mode).toBe('t-visual');
      
      // Press 'e' to move to word end at each mark independently
      pressKey('e');
      await pressKey('y');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.multiCursorClipboard.length).toBe(2);
      expect(editor.multiCursorClipboard[0]).toBe('tools');
      expect(editor.multiCursorClipboard[1]).toBe('hello');
      expect(editor.mode).toBe('normal');
    });

    it('should paste at tMark positions and enter TInsert mode', async () => {
      editor.setContent(['abc def', 'ghi jkl']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      // Add tMarks and yank
      editor.cursorX = 0;
      editor.cursorY = 0;
      pressKey('t');
      
      editor.cursorX = 0;
      editor.cursorY = 1;
      pressKey('t');
      
      // TVisual yank with 'e'
      editor.cursorX = 0;
      editor.cursorY = 0;
      pressKey('v');
      pressKey('e');
      await pressKey('y');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.multiCursorClipboard[0]).toBe('abc');
      expect(editor.multiCursorClipboard[1]).toBe('ghi');
      
      // Set new tMarks for pasting
      editor.tMarks = [
        { x: 4, y: 0 },
        { x: 4, y: 1 }
      ];
      
      // Paste should enter TInsert mode
      pressKey('p');
      
      expect(editor.mode).toBe('t-insert');
      expect(editor.content[0]).toContain('abc');
      expect(editor.content[1]).toContain('ghi');
    });

    it('should exit TVisual mode with Escape', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      editor.cursorX = 0;
      editor.cursorY = 0;
      pressKey('t');
      
      pressKey('v');
      expect(editor.mode).toBe('t-visual');
      
      pressKey('Escape');
      expect(editor.mode).toBe('normal');
    });
  });
});
