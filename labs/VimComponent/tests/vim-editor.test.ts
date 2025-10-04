import { describe, it, expect, beforeEach } from 'vitest';
import { VimEditor } from '../src/vim-editor';

describe('VimEditor', () => {
  let editor: VimEditor;

  beforeEach(() => {
    editor = document.createElement('vim-editor') as VimEditor;
    document.body.appendChild(editor);
    
    editor.mode = 'normal';
    editor.cursorX = 0;
    editor.cursorY = 0;
  });

  describe('$ key in normal mode', () => {
    it('should move cursor to end of line when content is "abc"', () => {
      editor.setContent(['abc']);
      
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
      expect(status.cursorX).toBe(0);
      expect(status.cursorY).toBe(0);
      
      const event = new KeyboardEvent('keydown', { key: '$' });
      window.dispatchEvent(event);
      
      const newStatus = editor.getStatus();
      expect(newStatus.cursorX).toBe(2);
      expect(newStatus.cursorY).toBe(0);
      
      const buffer = editor.getBuffer();
      expect(buffer[0][0].char).toBe('a');
      expect(buffer[0][1].char).toBe('b');
      expect(buffer[0][2].char).toBe('c');
      
      expect(buffer[0][2].background).toEqual([255, 255, 255]);
      expect(buffer[0][2].foreground).toEqual([0, 0, 0]);
    });

    it('should handle empty line', () => {
      editor.setContent(['']);
      
      const event = new KeyboardEvent('keydown', { key: '$' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(0);
    });

    it('should move to last character of long line', () => {
      editor.setContent(['hello world']);
      
      const event = new KeyboardEvent('keydown', { key: '$' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(10);
    });
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
      
      const buffer = editor.getBuffer();
      
      expect(buffer[0][1].background).toEqual([255, 255, 255]);
      expect(buffer[0][1].foreground).toEqual([0, 0, 0]);
      
      expect(buffer[0][0].background).toEqual([0, 0, 0]);
      expect(buffer[0][0].foreground).toEqual([255, 255, 255]);
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
});

