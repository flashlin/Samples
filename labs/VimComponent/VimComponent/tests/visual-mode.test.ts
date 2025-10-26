import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { setupP5Mock, createTestEditor, cleanupTestEditor, pressKey, pressKeys } from './test-helpers';

setupP5Mock();

import '../src/vim-editor';

describe('VimEditor - Visual Mode', () => {
  let editor: any;

  beforeEach(async () => {
    editor = await createTestEditor();
  });

  afterEach(() => {
    cleanupTestEditor(editor);
  });

  describe('visual mode navigation', () => {
    it('should move cursor to line end with $ key', () => {
      editor.setContent(['hello world']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('v');
      
      pressKey('$');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(10);
      expect(status.mode).toBe('visual');
    });

    it('should move cursor to line start with ^ key', () => {
      editor.setContent(['  hello world']);
      editor.cursorX = 8;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('v');
      
      pressKey('^');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(2);
      expect(status.mode).toBe('visual');
    });

    it('should select from start to end with $ key', () => {
      editor.setContent(['abc']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('v');
      
      pressKey('$');
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(2);
      
      editor.updateBuffer();
      const buffer = editor.getBuffer();
      
      for (let i = 0; i <= 2; i++) {
        expect(buffer[0][i].background).toEqual([100, 149, 237]);
      }
    });

    it('should support number prefix like 5j in visual mode', () => {
      editor.setContent(['line1', 'line2', 'line3', 'line4', 'line5', 'line6', 'line7']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('v');
      
      pressKey('5');
      
      pressKey('j');
      
      const status = editor.getStatus();
      expect(status.cursorY).toBe(5);
      expect(status.mode).toBe('visual');
      
      editor.updateBuffer();
      const buffer = editor.getBuffer();
      
      for (let y = 0; y <= 5; y++) {
        expect(buffer[y][0].background).toEqual([100, 149, 237]);
      }
    });

    it('should support number prefix like 3k in visual mode', () => {
      editor.setContent(['line1', 'line2', 'line3', 'line4', 'line5', 'line6', 'line7']);
      editor.cursorX = 0;
      editor.cursorY = 6;
      editor.mode = 'normal';
      
      pressKey('v');
      
      pressKey('3');
      
      pressKey('k');
      
      const status = editor.getStatus();
      expect(status.cursorY).toBe(3);
      expect(status.mode).toBe('visual');
    });

    it('should delete selected text with x key in visual mode', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      editor.resetHistory();
      
      pressKey('v');
      await editor.updateComplete;
      
      editor.cursorX = 4;
      await editor.updateComplete;
      
      await pressKey('x');
      await editor.updateComplete;
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content[0]).toBe(' world');
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
    });

    it('should delete multiline selection with x key in visual mode', async () => {
      editor.setContent(['line1', 'line2', 'line3', 'line4']);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 1;
      editor.mode = 'normal';
      editor.resetHistory();
      
      pressKey('v');
      await editor.updateComplete;
      
      pressKey('j');
      await editor.updateComplete;
      
      pressKey('j');
      await editor.updateComplete;
      
      await pressKey('x');
      await editor.updateComplete;
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content).toEqual(['line1', 'lie4']);
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
      expect(status.cursorY).toBe(1);
    });
  });

  describe('visual line mode', () => {
    it('should enter visual-line mode with V key', () => {
      editor.setContent(['line1', 'line2', 'line3']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('V');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('visual-line');
    });

    it('should highlight entire line in visual-line mode', () => {
      editor.setContent(['hello world']);
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('V');
      
      editor.updateBuffer();
      const buffer = editor.getBuffer();
      
      for (let x = 0; x < 11; x++) {
        expect(buffer[0][x].background).toEqual([100, 149, 237]);
      }
    });

    it('should highlight multiple lines when moving down', () => {
      editor.setContent(['line1', 'line2', 'line3', 'line4']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('V');
      
      pressKey('j');
      
      pressKey('j');
      
      const status = editor.getStatus();
      expect(status.cursorY).toBe(2);
      expect(status.mode).toBe('visual-line');
      
      editor.updateBuffer();
      const buffer = editor.getBuffer();
      
      for (let y = 0; y <= 2; y++) {
        for (let x = 0; x < 5; x++) {
          expect(buffer[y][x].background).toEqual([100, 149, 237]);
        }
      }
    });

    it('should support number prefix like 3j in visual-line mode', () => {
      editor.setContent(['line1', 'line2', 'line3', 'line4', 'line5']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('V');
      
      pressKey('3');
      
      pressKey('j');
      
      const status = editor.getStatus();
      expect(status.cursorY).toBe(3);
      expect(status.mode).toBe('visual-line');
    });

    it('should delete entire lines with x key in visual-line mode', async () => {
      editor.setContent(['line1', 'line2', 'line3', 'line4', 'line5']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 1;
      editor.mode = 'normal';
      editor.resetHistory();
      
      pressKey('V');
      await editor.updateComplete;
      
      pressKey('j');
      await editor.updateComplete;
      
      await pressKey('x');
      await editor.updateComplete;
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content).toEqual(['line1', 'line4', 'line5']);
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
      expect(status.cursorY).toBe(1);
    });

    it('should delete single line with x key in visual-line mode', async () => {
      editor.setContent(['line1', 'line2', 'line3']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 1;
      editor.mode = 'normal';
      editor.resetHistory();
      
      pressKey('V');
      await editor.updateComplete;
      
      await pressKey('x');
      await editor.updateComplete;
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content).toEqual(['line1', 'line3']);
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
      expect(status.cursorY).toBe(1);
    });

    it('should support line-wise paste with P (paste before) after cutting lines', async () => {
      editor.setContent(['line1', 'line2', 'line3', 'line4']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 1;
      editor.mode = 'normal';
      editor.resetHistory();
      
      // Select line2
      pressKey('V');
      await editor.updateComplete;
      
      // Cut line2 (should add trailing newline to clipboard)
      await pressKey('x');
      await editor.updateComplete;
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content).toEqual(['line1', 'line3', 'line4']);
      expect(editor.cursorY).toBe(1);
      
      // Move to line1
      pressKey('k');
      await editor.updateComplete;
      expect(editor.cursorY).toBe(0);
      
      // Paste before line1 (should insert as complete line)
      await pressKey('P');
      await editor.updateComplete;
      await new Promise(resolve => setTimeout(resolve, 50));
      
      // line2 should be inserted before line1
      expect(editor.content).toEqual(['line2', 'line1', 'line3', 'line4']);
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(0);
    });

    it('should support line-wise paste with p (paste after) after cutting lines', async () => {
      editor.setContent(['line1', 'line2', 'line3', 'line4']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 1;
      editor.mode = 'normal';
      editor.resetHistory();
      
      // Select line2
      pressKey('V');
      await editor.updateComplete;
      
      // Cut line2
      await pressKey('x');
      await editor.updateComplete;
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content).toEqual(['line1', 'line3', 'line4']);
      expect(editor.cursorY).toBe(1);
      
      // Move to line1
      pressKey('k');
      await editor.updateComplete;
      expect(editor.cursorY).toBe(0);
      
      // Paste after line1 (should insert as complete line)
      await pressKey('p');
      await editor.updateComplete;
      await new Promise(resolve => setTimeout(resolve, 50));
      
      // line2 should be inserted after line1
      expect(editor.content).toEqual(['line1', 'line2', 'line3', 'line4']);
      expect(editor.cursorY).toBe(1);
      expect(editor.cursorX).toBe(0);
    });

    it('should support line-wise paste with multiple lines', async () => {
      editor.setContent(['line1', 'line2', 'line3', 'line4', 'line5']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 1;
      editor.mode = 'normal';
      editor.resetHistory();
      
      // Select line2 and line3
      pressKey('V');
      await editor.updateComplete;
      pressKey('j');
      await editor.updateComplete;
      
      // Cut line2 and line3
      await pressKey('x');
      await editor.updateComplete;
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content).toEqual(['line1', 'line4', 'line5']);
      expect(editor.cursorY).toBe(1);
      
      // Move to line5
      pressKey('j');
      await editor.updateComplete;
      expect(editor.cursorY).toBe(2);
      
      // Paste after line5
      await pressKey('p');
      await editor.updateComplete;
      await new Promise(resolve => setTimeout(resolve, 50));
      
      // line2 and line3 should be inserted after line5
      expect(editor.content).toEqual(['line1', 'line4', 'line5', 'line2', 'line3']);
      expect(editor.cursorY).toBe(3);
      expect(editor.cursorX).toBe(0);
    });

    it('should restore original content with V-x-P sequence', async () => {
      editor.setContent(['line1', 'line2', 'line3']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 1;
      editor.mode = 'normal';
      editor.resetHistory();
      
      // Select line2
      pressKey('V');
      await editor.updateComplete;
      
      // Cut line2
      await pressKey('x');
      await editor.updateComplete;
      await new Promise(resolve => setTimeout(resolve, 50));
      
      // Content should be: line1, line3
      expect(editor.content).toEqual(['line1', 'line3']);
      
      // Paste before current line (line3 became line2)
      await pressKey('P');
      await editor.updateComplete;
      await new Promise(resolve => setTimeout(resolve, 50));
      
      // Original content should be restored
      expect(editor.content).toEqual(['line1', 'line2', 'line3']);
      expect(editor.cursorY).toBe(1);
    });
  });
});
