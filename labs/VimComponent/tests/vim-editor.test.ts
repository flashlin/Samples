import { describe, it, expect, beforeEach, vi } from 'vitest';
import '../src/vim-editor';

const mockRedraw = vi.fn();
const mockP5Constructor = vi.fn((sketch: any, element: any) => {
  const mockP5Instance = {
    setup: vi.fn(),
    draw: vi.fn(),
    createCanvas: vi.fn(() => ({
      elt: document.createElement('canvas')
    })),
    background: vi.fn(),
    fill: vi.fn(),
    stroke: vi.fn(),
    noFill: vi.fn(),
    noStroke: vi.fn(),
    rect: vi.fn(),
    text: vi.fn(),
    textSize: vi.fn(),
    textAlign: vi.fn(),
    textFont: vi.fn(),
    textWidth: vi.fn(() => 9),
    noLoop: vi.fn(),
    redraw: mockRedraw,
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
});

(window as any).p5 = mockP5Constructor;

describe('VimEditor', () => {
  let editor: any;

  beforeEach(async () => {
    mockRedraw.mockClear();
    
    editor = document.createElement('vim-editor');
    document.body.appendChild(editor);
    
    await new Promise(resolve => setTimeout(resolve, 50));
    
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
      
      editor.updateBuffer();
      
      const buffer = editor.getBuffer();
      expect(buffer[0][0].char).toBe('a');
      expect(buffer[0][1].char).toBe('b');
      expect(buffer[0][2].char).toBe('c');
      
      if (newStatus.cursorVisible) {
        expect(buffer[0][2].background).toEqual([255, 255, 255]);
        expect(buffer[0][2].foreground).toEqual([0, 0, 0]);
      }
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

  describe('w key in normal mode', () => {
    it('should move cursor to next English word', () => {
      editor.setContent(['hello world']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(6);
      expect(status.cursorY).toBe(0);
    });

    it('should move cursor to next Chinese word group', () => {
      editor.setContent(['你好世界']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(3);
      expect(status.cursorY).toBe(0);
    });

    it('should move cursor between English and Chinese', () => {
      editor.setContent(['Hello 你好 World']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      let event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      let status = editor.getStatus();
      expect(status.cursorX).toBe(6);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      status = editor.getStatus();
      expect(status.cursorX).toBe(9);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      status = editor.getStatus();
      expect(status.cursorX).toBe(13);
    });

    it('should jump from punctuation to next word', () => {
      editor.setContent(['Hello World中文!']);
      editor.cursorX = 14;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(14);
      expect(status.cursorY).toBe(0);
    });

    it('should move to next line when at end of line', () => {
      editor.setContent(['hello', 'world']);
      editor.cursorX = 4;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(0);
      expect(status.cursorY).toBe(1);
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
});

