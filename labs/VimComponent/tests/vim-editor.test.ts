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

function pressKey(key: string) {
  const event = new KeyboardEvent('keydown', { key });
  window.dispatchEvent(event);
}

function pressKeys(...keys: string[]) {
  keys.forEach(key => pressKey(key));
}

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

  describe('W key in normal mode', () => {
    it('should move cursor to next space-separated WORD', () => {
      editor.setContent(['hello,world test']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: 'W' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(12);
      expect(status.cursorY).toBe(0);
    });

    it('should treat punctuation as part of WORD', () => {
      editor.setContent(['hello,world! test']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: 'W' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(13);
    });

    it('should move across Chinese and English together', () => {
      editor.setContent(['hello中文world test']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: 'W' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(13);
    });
  });

  describe('B key in normal mode', () => {
    it('should move cursor to previous space-separated WORD', () => {
      editor.setContent(['hello,world test']);
      editor.cursorX = 12;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: 'B' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(0);
      expect(status.cursorY).toBe(0);
    });

    it('should treat punctuation as part of WORD', () => {
      editor.setContent(['hello,world! test']);
      editor.cursorX = 13;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: 'B' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(0);
    });

    it('should move across Chinese and English together', () => {
      editor.setContent(['hello中文world test']);
      editor.cursorX = 13;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: 'B' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(0);
    });
  });

  describe('e key in normal mode', () => {
    it('should move to end of current English word', () => {
      editor.setContent(['hello world']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: 'e' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(4);
      expect(status.cursorY).toBe(0);
    });

    it('should move to end of current Chinese word', () => {
      editor.setContent(['你好世界']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: 'e' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(3);
    });

    it('should stay at same position when cursor is on space', () => {
      editor.setContent(['hello world']);
      editor.cursorX = 5;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: 'e' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(5);
    });

    it('should handle cursor in middle of word', () => {
      editor.setContent(['hello world']);
      editor.cursorX = 2;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: 'e' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(4);
    });

    it('should handle cursor at end of word', () => {
      editor.setContent(['hello world']);
      editor.cursorX = 4;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: 'e' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(10);
    });

    it('should handle mixed English and Chinese', () => {
      editor.setContent(['hello中文world']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: 'e' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(4);
    });

    it('should handle number prefix like 2e', () => {
      editor.setContent(['one two three four']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      let event = new KeyboardEvent('keydown', { key: '2' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'e' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(6);
    });
  });

  describe('^ key in normal mode', () => {
    it('should move cursor to first non-space character', () => {
      editor.setContent(['  hello world']);
      editor.cursorX = 8;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: '^' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(2);
      expect(status.cursorY).toBe(0);
    });

    it('should move to position 0 if no leading spaces', () => {
      editor.setContent(['hello world']);
      editor.cursorX = 8;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: '^' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(0);
    });

    it('should handle line with only spaces', () => {
      editor.setContent(['     ']);
      editor.cursorX = 3;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: '^' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(0);
    });

    it('should handle tabs and spaces', () => {
      editor.setContent(['\t  hello']);
      editor.cursorX = 5;
      editor.cursorY = 0;
      
      const event = new KeyboardEvent('keydown', { key: '^' });
      window.dispatchEvent(event);
      
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
      
      let event = new KeyboardEvent('keydown', { key: '5' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'j' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorY).toBe(5);
    });

    it('should move up 3 lines with 3k', () => {
      editor.setContent(['line1', 'line2', 'line3', 'line4', 'line5', 'line6', 'line7']);
      editor.cursorX = 0;
      editor.cursorY = 6;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: '3' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'k' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorY).toBe(3);
    });

    it('should support multi-digit numbers like 10j', () => {
      const lines = Array.from({ length: 20 }, (_, i) => `line${i + 1}`);
      editor.setContent(lines);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: '1' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: '0' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'j' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorY).toBe(10);
    });

    it('should work with other movement keys like 5w', () => {
      editor.setContent(['one two three four five six seven']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: '5' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(24);
    });
  });

  describe('visual mode navigation', () => {
    it('should move cursor to line end with $ key', () => {
      editor.setContent(['hello world']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'v' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: '$' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(10);
      expect(status.mode).toBe('visual');
    });

    it('should move cursor to line start with ^ key', () => {
      editor.setContent(['  hello world']);
      editor.cursorX = 8;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'v' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: '^' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(2);
      expect(status.mode).toBe('visual');
    });

    it('should select from start to end with $ key', () => {
      editor.setContent(['abc']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'v' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: '$' });
      window.dispatchEvent(event);
      
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
      
      let event = new KeyboardEvent('keydown', { key: 'v' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: '5' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'j' });
      window.dispatchEvent(event);
      
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
      
      let event = new KeyboardEvent('keydown', { key: 'v' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: '3' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'k' });
      window.dispatchEvent(event);
      
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
      
      let event = new KeyboardEvent('keydown', { key: 'v' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
      editor.cursorX = 4;
      await editor.updateComplete;
      
      event = new KeyboardEvent('keydown', { key: 'x' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
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
      
      let event = new KeyboardEvent('keydown', { key: 'v' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
      event = new KeyboardEvent('keydown', { key: 'j' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
      event = new KeyboardEvent('keydown', { key: 'j' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
      event = new KeyboardEvent('keydown', { key: 'x' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
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
      
      const event = new KeyboardEvent('keydown', { key: 'V' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.mode).toBe('visual-line');
    });

    it('should highlight entire line in visual-line mode', () => {
      editor.setContent(['hello world']);
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      const event = new KeyboardEvent('keydown', { key: 'V' });
      window.dispatchEvent(event);
      
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
      
      let event = new KeyboardEvent('keydown', { key: 'V' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'j' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'j' });
      window.dispatchEvent(event);
      
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
      
      let event = new KeyboardEvent('keydown', { key: 'V' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: '3' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'j' });
      window.dispatchEvent(event);
      
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
      
      let event = new KeyboardEvent('keydown', { key: 'V' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
      event = new KeyboardEvent('keydown', { key: 'j' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
      event = new KeyboardEvent('keydown', { key: 'x' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
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
      
      let event = new KeyboardEvent('keydown', { key: 'V' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
      event = new KeyboardEvent('keydown', { key: 'x' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
      expect(editor.content).toEqual(['line1', 'line3']);
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
      expect(status.cursorY).toBe(1);
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
        const event = new KeyboardEvent('keydown', { key: 'j' });
        window.dispatchEvent(event);
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
        const event = new KeyboardEvent('keydown', { key: 'j' });
        window.dispatchEvent(event);
      }
      
      for (let i = 0; i < 60; i++) {
        const event = new KeyboardEvent('keydown', { key: 'k' });
        window.dispatchEvent(event);
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
        const event = new KeyboardEvent('keydown', { key: 'l' });
        window.dispatchEvent(event);
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
        const event = new KeyboardEvent('keydown', { key: 'h' });
        window.dispatchEvent(event);
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
      
      const event = new KeyboardEvent('keydown', { key: '$' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.cursorX).toBe(99);
      
      const scroll = editor.getScrollOffset();
      expect(scroll.x).toBeGreaterThan(0);
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

  describe('fast-jump mode', () => {
    it('should enter fast-jump mode when pressing f', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      
      const event = new KeyboardEvent('keydown', { key: 'f' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.mode).toBe('fast-jump');
    });

    it('should return to normal mode if no matches found', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      
      let event = new KeyboardEvent('keydown', { key: 'f' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'x' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
    });

    it('should jump directly if only one match found', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      let event = new KeyboardEvent('keydown', { key: 'f' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
      expect(status.cursorX).toBe(6);
      expect(status.cursorY).toBe(0);
    });

    it('should enter match mode if multiple matches found', async () => {
      editor.setContent(['hello hello hello']);
      await editor.updateComplete;
      
      let event = new KeyboardEvent('keydown', { key: 'f' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'h' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.mode).toBe('match');
    });

    it('should jump to correct position after selecting label', async () => {
      editor.setContent(['hello hello hello']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      let event = new KeyboardEvent('keydown', { key: 'f' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'h' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'b' });
      window.dispatchEvent(event);
      
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
      expect(status.mode).toBe('match');
      
      pressKey('b');
      
      status = editor.getStatus();
      expect(status.mode).toBe('normal');
      expect(status.cursorX).toBe(1);
      expect(status.cursorY).toBe(0);
    });

    it('should exit on Escape in fast-jump mode', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      
      let event = new KeyboardEvent('keydown', { key: 'f' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'Escape' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
    });

    it('should exit on Escape in match mode', async () => {
      editor.setContent(['hello hello hello']);
      await editor.updateComplete;
      
      let event = new KeyboardEvent('keydown', { key: 'f' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'h' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'Escape' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
    });

    it('should work in visual mode', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      
      let event = new KeyboardEvent('keydown', { key: 'v' });
      window.dispatchEvent(event);
      
      expect(editor.getStatus().mode).toBe('visual');
      
      event = new KeyboardEvent('keydown', { key: 'f' });
      window.dispatchEvent(event);
      
      expect(editor.getStatus().mode).toBe('fast-jump');
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
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
      
      let event = new KeyboardEvent('keydown', { key: 'V' });
      window.dispatchEvent(event);
      
      expect(editor.getStatus().mode).toBe('visual-line');
      
      event = new KeyboardEvent('keydown', { key: 'f' });
      window.dispatchEvent(event);
      
      expect(editor.getStatus().mode).toBe('fast-jump');
      
      event = new KeyboardEvent('keydown', { key: 'b' });
      window.dispatchEvent(event);
      
      expect(editor.getStatus().mode).toBe('match');
      
      event = new KeyboardEvent('keydown', { key: 'b' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.mode).toBe('visual-line');
      expect(status.cursorX).toBe(8);
      expect(status.cursorY).toBe(1);
    });

    it('should return to visual mode when pressing Escape in fast-jump', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      
      let event = new KeyboardEvent('keydown', { key: 'v' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'f' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'Escape' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.mode).toBe('visual');
    });

    it('should return to visual mode when pressing Escape in match mode', async () => {
      editor.setContent(['hello hello hello']);
      await editor.updateComplete;
      
      let event = new KeyboardEvent('keydown', { key: 'v' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'f' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'h' });
      window.dispatchEvent(event);
      
      expect(editor.getStatus().mode).toBe('match');
      
      event = new KeyboardEvent('keydown', { key: 'Escape' });
      window.dispatchEvent(event);
      
      const status = editor.getStatus();
      expect(status.mode).toBe('visual');
    });
  });

  describe('diw command', () => {
    it('should delete English word under cursor', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 7;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'i' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      expect(editor.content[0]).toBe('hello  test');
      expect(editor.cursorX).toBe(6);
    });

    it('should delete Chinese word under cursor', async () => {
      editor.setContent(['你好世界測試']);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'i' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      expect(editor.content[0]).toBe('');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete word with numbers', async () => {
      editor.setContent(['hello world123 test']);
      await editor.updateComplete;
      editor.cursorX = 8;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'i' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      expect(editor.content[0]).toBe('hello  test');
      expect(editor.cursorX).toBe(6);
    });

    it('should delete word at start of line', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'i' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      expect(editor.content[0]).toBe(' world');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete word at end of line', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorX = 10;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'i' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      expect(editor.content[0]).toBe('hello ');
      expect(editor.cursorX).toBe(5);
    });

    it('should not delete when cursor is on space', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'i' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      expect(editor.content[0]).toBe('hello world');
    });

    it('should handle mixed English and Chinese', async () => {
      editor.setContent(['hello你好world']);
      await editor.updateComplete;
      editor.cursorX = 6;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'i' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      expect(editor.content[0]).toBe('helloworld');
      expect(editor.cursorX).toBe(5);
    });
  });

  describe('di` command (backtick)', () => {
    it('should delete content between backticks', async () => {
      editor.setContent(['const str = `hello world`;']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', '`');
      
      expect(editor.content[0]).toBe('const str = ``;');
      expect(editor.cursorX).toBe(13);
    });

    it('should handle cursor on opening backtick', async () => {
      editor.setContent(['`test content`']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', '`');
      
      expect(editor.content[0]).toBe('``');
      expect(editor.cursorX).toBe(1);
    });

    it('should handle cursor on closing backtick', async () => {
      editor.setContent(['`test content`']);
      await editor.updateComplete;
      editor.cursorX = 13;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', '`');
      
      expect(editor.content[0]).toBe('``');
      expect(editor.cursorX).toBe(1);
    });

    it('should do nothing if no matching backticks', async () => {
      editor.setContent(['no backticks here']);
      await editor.updateComplete;
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', '`');
      
      expect(editor.content[0]).toBe('no backticks here');
      expect(editor.cursorX).toBe(5);
    });

    it('should handle empty content between backticks', async () => {
      editor.setContent(['test `` string']);
      await editor.updateComplete;
      editor.cursorX = 6;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', '`');
      
      expect(editor.content[0]).toBe('test `` string');
      expect(editor.cursorX).toBe(6);
    });
  });

  describe("di' command (single quote)", () => {
    it('should delete content between single quotes', async () => {
      editor.setContent(["const str = 'hello world';"]);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', "'");
      
      expect(editor.content[0]).toBe("const str = '';");
      expect(editor.cursorX).toBe(13);
    });

    it('should handle escaped single quote', async () => {
      editor.setContent(["const str = 'don\\'t worry';"]);
      await editor.updateComplete;
      editor.cursorX = 18;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', "'");
      
      expect(editor.content[0]).toBe("const str = '';");
      expect(editor.cursorX).toBe(13);
    });

    it('should handle multiple escaped backslashes', async () => {
      editor.setContent(["const str = 'test\\\\\\' string';"]);
      await editor.updateComplete;
      editor.cursorX = 20;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', "'");
      
      expect(editor.content[0]).toBe("const str = '';");
      expect(editor.cursorX).toBe(13);
    });

    it('should do nothing if no matching quotes', async () => {
      editor.setContent(['no quotes here']);
      await editor.updateComplete;
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', "'");
      
      expect(editor.content[0]).toBe('no quotes here');
      expect(editor.cursorX).toBe(5);
    });
  });

  describe('di" command (double quote)', () => {
    it('should delete content between double quotes', async () => {
      editor.setContent(['const str = "hello world";']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', '"');
      
      expect(editor.content[0]).toBe('const str = "";');
      expect(editor.cursorX).toBe(13);
    });

    it('should handle escaped double quote', async () => {
      editor.setContent(['const str = "say \\"hello\\"";']);
      await editor.updateComplete;
      editor.cursorX = 20;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', '"');
      
      expect(editor.content[0]).toBe('const str = "";');
      expect(editor.cursorX).toBe(13);
    });

    it('should handle multiple escaped backslashes before quote', async () => {
      editor.setContent(['const str = "test\\\\\\" string";']);
      await editor.updateComplete;
      editor.cursorX = 22;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', '"');
      
      expect(editor.content[0]).toBe('const str = "";');
      expect(editor.cursorX).toBe(13);
    });

    it('should handle cursor on opening quote', async () => {
      editor.setContent(['"test content"']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', '"');
      
      expect(editor.content[0]).toBe('""');
      expect(editor.cursorX).toBe(1);
    });

    it('should do nothing if no matching quotes', async () => {
      editor.setContent(['no quotes here']);
      await editor.updateComplete;
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', '"');
      
      expect(editor.content[0]).toBe('no quotes here');
      expect(editor.cursorX).toBe(5);
    });
  });

  describe('dw command', () => {
    it('should delete word from cursor position', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      expect(editor.content[0]).toBe('world test');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete from middle of word', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      expect(editor.content[0]).toBe('heworld test');
      expect(editor.cursorX).toBe(2);
    });

    it('should delete Chinese word from cursor', async () => {
      editor.setContent(['你好世界測試']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      expect(editor.content[0]).toBe('');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete word including trailing spaces', async () => {
      editor.setContent(['hello  world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      expect(editor.content[0]).toBe('world');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete spaces when cursor on space', async () => {
      editor.setContent(['hello  world']);
      await editor.updateComplete;
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      expect(editor.content[0]).toBe('helloworld');
      expect(editor.cursorX).toBe(5);
    });

    it('should handle word with numbers', async () => {
      editor.setContent(['hello123 world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      expect(editor.content[0]).toBe('world');
      expect(editor.cursorX).toBe(0);
    });

    it('should handle delete at end of line', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorX = 6;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      expect(editor.content[0]).toBe('hello ');
      expect(editor.cursorX).toBe(5);
    });

    it('should handle mixed English and Chinese with dw', async () => {
      editor.setContent(['hello你好world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      
      expect(editor.content[0]).toBe('你好world');
      expect(editor.cursorX).toBe(0);
    });
  });

  describe('de command', () => {
    it('should delete to end of word from cursor position', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'e');
      
      expect(editor.content[0]).toBe(' world test');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete to end of word from middle', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'e');
      
      expect(editor.content[0]).toBe('he world test');
      expect(editor.cursorX).toBe(2);
    });

    it('should delete Chinese word to end', async () => {
      editor.setContent(['你好世界測試']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'e');
      
      expect(editor.content[0]).toBe('');
      expect(editor.cursorX).toBe(0);
    });

    it('should not delete trailing spaces', async () => {
      editor.setContent(['hello  world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'e');
      
      expect(editor.content[0]).toBe('  world');
      expect(editor.cursorX).toBe(0);
    });

    it('should do nothing when cursor on space', async () => {
      editor.setContent(['hello  world']);
      await editor.updateComplete;
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'e');
      
      expect(editor.content[0]).toBe('hello  world');
      expect(editor.cursorX).toBe(5);
    });

    it('should handle word with numbers', async () => {
      editor.setContent(['hello123 world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'e');
      
      expect(editor.content[0]).toBe(' world');
      expect(editor.cursorX).toBe(0);
    });

    it('should handle delete at end of line', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorX = 6;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'e');
      
      expect(editor.content[0]).toBe('hello ');
      expect(editor.cursorX).toBe(5);
    });

    it('should handle mixed English and Chinese', async () => {
      editor.setContent(['hello你好world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'e');
      
      expect(editor.content[0]).toBe('你好world');
      expect(editor.cursorX).toBe(0);
    });

    it('should do nothing on empty line', async () => {
      editor.setContent(['']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'e');
      
      expect(editor.content[0]).toBe('');
      expect(editor.cursorX).toBe(0);
    });

    it('should handle cursor at last character', async () => {
      editor.setContent(['hello']);
      await editor.updateComplete;
      editor.cursorX = 4;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'e');
      
      expect(editor.content[0]).toBe('hell');
      expect(editor.cursorX).toBe(3);
    });
  });

  describe('p command (paste)', () => {
    let mockReadText: any;
    let mockWriteText: any;

    beforeEach(() => {
      mockReadText = vi.fn();
      mockWriteText = vi.fn();
      
      Object.defineProperty(navigator, 'clipboard', {
        value: {
          readText: mockReadText,
          writeText: mockWriteText,
        },
        writable: true,
        configurable: true,
      });
    });

    it('should paste single line text after cursor', async () => {
      editor.content = ['hello world'];
      await editor.updateComplete;
      editor.cursorX = 4;
      editor.cursorY = 0;
      editor.mode = 'normal';

      mockReadText.mockResolvedValue('TEST');

      const event = new KeyboardEvent('keydown', { key: 'p' });
      window.dispatchEvent(event);

      await new Promise(resolve => setTimeout(resolve, 10));
      await editor.updateComplete;

      expect(editor.content[0]).toBe('helloTEST world');
      expect(editor.cursorX).toBe(8);
      expect(editor.cursorY).toBe(0);
    });

    it('should paste at beginning of line', async () => {
      editor.content = ['hello'];
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';

      mockReadText.mockResolvedValue('X');

      const event = new KeyboardEvent('keydown', { key: 'p' });
      window.dispatchEvent(event);

      await new Promise(resolve => setTimeout(resolve, 10));
      await editor.updateComplete;

      expect(editor.content[0]).toBe('hXello');
      expect(editor.cursorX).toBe(1);
    });

    it('should paste at end of line', async () => {
      editor.content = ['hello'];
      await editor.updateComplete;
      editor.cursorX = 4;
      editor.cursorY = 0;
      editor.mode = 'normal';

      mockReadText.mockResolvedValue('!');

      const event = new KeyboardEvent('keydown', { key: 'p' });
      window.dispatchEvent(event);

      await new Promise(resolve => setTimeout(resolve, 10));
      await editor.updateComplete;

      expect(editor.content[0]).toBe('hello!');
      expect(editor.cursorX).toBe(5);
    });

    it('should paste multi-line text after cursor', async () => {
      editor.content = ['hello world'];
      await editor.updateComplete;
      editor.cursorX = 4;
      editor.cursorY = 0;
      editor.mode = 'normal';

      mockReadText.mockResolvedValue('AAA\nBBB\nCCC');

      const event = new KeyboardEvent('keydown', { key: 'p' });
      window.dispatchEvent(event);

      await new Promise(resolve => setTimeout(resolve, 10));
      await editor.updateComplete;

      expect(editor.content[0]).toBe('helloAAA');
      expect(editor.content[1]).toBe('BBB');
      expect(editor.content[2]).toBe('CCC world');
      expect(editor.cursorX).toBe(2);
      expect(editor.cursorY).toBe(2);
    });

    it('should paste Chinese text after cursor', async () => {
      editor.content = ['你好世界'];
      await editor.updateComplete;
      editor.cursorX = 1;
      editor.cursorY = 0;
      editor.mode = 'normal';

      mockReadText.mockResolvedValue('測試');

      const event = new KeyboardEvent('keydown', { key: 'p' });
      window.dispatchEvent(event);

      await new Promise(resolve => setTimeout(resolve, 10));
      await editor.updateComplete;

      expect(editor.content[0]).toBe('你好測試世界');
      expect(editor.cursorX).toBe(3);
    });

    it('should paste on empty line', async () => {
      editor.content = [''];
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';

      mockReadText.mockResolvedValue('test');

      const event = new KeyboardEvent('keydown', { key: 'p' });
      window.dispatchEvent(event);

      await new Promise(resolve => setTimeout(resolve, 10));
      await editor.updateComplete;

      expect(editor.content[0]).toBe('test');
      expect(editor.cursorX).toBe(3);
    });

    it('should paste multiple lines on empty line', async () => {
      editor.content = [''];
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';

      mockReadText.mockResolvedValue('line1\nline2\nline3');

      const event = new KeyboardEvent('keydown', { key: 'p' });
      window.dispatchEvent(event);

      await new Promise(resolve => setTimeout(resolve, 10));
      await editor.updateComplete;

      expect(editor.content[0]).toBe('line1');
      expect(editor.content[1]).toBe('line2');
      expect(editor.content[2]).toBe('line3');
      expect(editor.cursorX).toBe(4);
      expect(editor.cursorY).toBe(2);
    });
  });

  describe('u command (undo)', () => {
    it('should have history initialized', async () => {
      editor.content = ['hello'];
      await editor.updateComplete;
      editor.resetHistory();
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';

      const event = new KeyboardEvent('keydown', { key: 'u' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content[0]).toBe('hello');
    });

    it('should undo diw command', async () => {
      editor.content = ['hello world test'];
      await editor.updateComplete;
      editor.cursorX = 6;
      editor.cursorY = 0;
      editor.mode = 'normal';
      editor.resetHistory();

      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);

      event = new KeyboardEvent('keydown', { key: 'i' });
      window.dispatchEvent(event);

      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content[0]).toBe('hello  test');

      event = new KeyboardEvent('keydown', { key: 'u' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content[0]).toBe('hello world test');
      expect(editor.cursorX).toBe(6);
    });

    it('should undo dw command', async () => {
      editor.content = ['hello world'];
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      editor.resetHistory();

      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);

      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content[0]).toBe('world');

      event = new KeyboardEvent('keydown', { key: 'u' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content[0]).toBe('hello world');
      expect(editor.cursorX).toBe(0);
    });

    it('should undo paste operation', async () => {
      const mockReadText = vi.fn();
      const mockWriteText = vi.fn();
      
      Object.defineProperty(navigator, 'clipboard', {
        value: {
          readText: mockReadText,
          writeText: mockWriteText,
        },
        writable: true,
        configurable: true,
      });

      editor.content = ['hello'];
      await editor.updateComplete;
      editor.cursorX = 4;
      editor.cursorY = 0;
      editor.mode = 'normal';
      editor.resetHistory();

      mockReadText.mockResolvedValue(' world');

      let event = new KeyboardEvent('keydown', { key: 'p' });
      window.dispatchEvent(event);

      await new Promise(resolve => setTimeout(resolve, 10));
      await editor.updateComplete;

      expect(editor.content[0]).toBe('hello world');

      event = new KeyboardEvent('keydown', { key: 'u' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content[0]).toBe('hello');
      expect(editor.cursorX).toBe(4);
    });

    it('should undo visual mode cut', async () => {
      editor.content = ['hello world'];
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      editor.resetHistory();

      let event = new KeyboardEvent('keydown', { key: 'v' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      editor.cursorX = 4;

      event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content[0]).toBe(' world');

      event = new KeyboardEvent('keydown', { key: 'u' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content[0]).toBe('hello world');
      expect(editor.cursorX).toBe(4);
    });

    it('should handle multiple undo operations', async () => {
      editor.content = ['hello world'];
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      editor.resetHistory();

      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      event = new KeyboardEvent('keydown', { key: 'i' });
      window.dispatchEvent(event);
      event = new KeyboardEvent('keydown', { key: 'w' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content[0]).toBe(' world');

      event = new KeyboardEvent('keydown', { key: 'u' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content[0]).toBe('hello world');
    });
  });

  describe('d{number}j and d{number}k commands', () => {
    it('should delete multiple lines down with d2j', async () => {
      editor.content = ['line1', 'line2', 'line3', 'line4', 'line5'];
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 1;
      editor.mode = 'normal';
      editor.resetHistory();

      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      event = new KeyboardEvent('keydown', { key: '2' });
      window.dispatchEvent(event);
      event = new KeyboardEvent('keydown', { key: 'j' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content).toEqual(['line1', 'line5']);
      expect(editor.cursorY).toBe(1);
      expect(editor.cursorX).toBe(0);
    });

    it('should delete multiple lines up with d2k', async () => {
      editor.content = ['line1', 'line2', 'line3', 'line4', 'line5'];
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 3;
      editor.mode = 'normal';
      editor.resetHistory();

      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      event = new KeyboardEvent('keydown', { key: '2' });
      window.dispatchEvent(event);
      event = new KeyboardEvent('keydown', { key: 'k' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content).toEqual(['line1', 'line5']);
      expect(editor.cursorY).toBe(1);
      expect(editor.cursorX).toBe(0);
    });

    it('should delete single line with d1j', async () => {
      editor.content = ['line1', 'line2', 'line3'];
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      editor.resetHistory();

      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      event = new KeyboardEvent('keydown', { key: '1' });
      window.dispatchEvent(event);
      event = new KeyboardEvent('keydown', { key: 'j' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content).toEqual(['line3']);
      expect(editor.cursorY).toBe(0);
    });

    it('should handle d5j when there are fewer lines', async () => {
      editor.content = ['line1', 'line2', 'line3'];
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      editor.resetHistory();

      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      event = new KeyboardEvent('keydown', { key: '5' });
      window.dispatchEvent(event);
      event = new KeyboardEvent('keydown', { key: 'j' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content).toEqual(['']);
      expect(editor.cursorY).toBe(0);
    });

    it('should handle d5k from top lines', async () => {
      editor.content = ['line1', 'line2', 'line3'];
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 1;
      editor.mode = 'normal';
      editor.resetHistory();

      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      event = new KeyboardEvent('keydown', { key: '5' });
      window.dispatchEvent(event);
      event = new KeyboardEvent('keydown', { key: 'k' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content).toEqual(['line3']);
      expect(editor.cursorY).toBe(0);
    });

    it('should undo d3j command', async () => {
      editor.content = ['line1', 'line2', 'line3', 'line4', 'line5', 'line6'];
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 1;
      editor.mode = 'normal';
      editor.resetHistory();

      let event = new KeyboardEvent('keydown', { key: 'd' });
      window.dispatchEvent(event);
      event = new KeyboardEvent('keydown', { key: '3' });
      window.dispatchEvent(event);
      event = new KeyboardEvent('keydown', { key: 'j' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content).toEqual(['line1', 'line6']);

      event = new KeyboardEvent('keydown', { key: 'u' });
      window.dispatchEvent(event);
      await editor.updateComplete;

      expect(editor.content).toEqual(['line1', 'line2', 'line3', 'line4', 'line5', 'line6']);
      expect(editor.cursorY).toBe(1);
    });
  });
});

