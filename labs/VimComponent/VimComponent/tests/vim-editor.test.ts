import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('p5', () => {
  return {
    default: vi.fn((sketch: any, element: any) => {
      const mockCanvas = {
        elt: document.createElement('canvas'),
        parent: vi.fn(),
      };
      
      const mockP5Instance = {
        setup: vi.fn(),
        draw: vi.fn(),
        createCanvas: vi.fn(() => mockCanvas),
        background: vi.fn(),
        fill: vi.fn(),
        stroke: vi.fn(),
        strokeWeight: vi.fn(),
        line: vi.fn(),
        noFill: vi.fn(),
        noStroke: vi.fn(),
        rect: vi.fn(),
        text: vi.fn(),
        textSize: vi.fn(),
        textAlign: vi.fn(),
        textFont: vi.fn(),
        textWidth: vi.fn(() => 9),
        noLoop: vi.fn(),
        loop: vi.fn(),
        redraw: vi.fn(),
        remove: vi.fn(),
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
    }),
  };
});

import '../src/vim-editor';

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

  describe('a command (append after cursor)', () => {
    it('should move cursor one position right and enter insert mode', async () => {
      editor.setContent(['hello world']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const event = new KeyboardEvent('keydown', { key: 'a' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
      const status = editor.getStatus();
      expect(status.mode).toBe('insert');
      expect(status.cursorX).toBe(1);
      expect(status.cursorY).toBe(0);
    });

    it('should NOT insert "a" character when entering insert mode', async () => {
      editor.setContent(['hello world']);
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const event = new KeyboardEvent('keydown', { key: 'a' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('hello world');
      expect(editor.cursorX).toBe(6);
      const status = editor.getStatus();
      expect(status.mode).toBe('insert');
    });

    it('should work at end of line', async () => {
      editor.setContent(['hello']);
      editor.cursorX = 4;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const event = new KeyboardEvent('keydown', { key: 'a' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
      expect(editor.cursorX).toBe(5);
      expect(editor.content[0]).toBe('hello');
      const status = editor.getStatus();
      expect(status.mode).toBe('insert');
    });

    it('should work on empty line', async () => {
      editor.setContent(['']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const event = new KeyboardEvent('keydown', { key: 'a' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
      expect(editor.cursorX).toBe(0);
      expect(editor.content[0]).toBe('');
      const status = editor.getStatus();
      expect(status.mode).toBe('insert');
    });

    it('should allow typing after pressing a', async () => {
      editor.setContent(['test']);
      editor.cursorX = 1;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const aEvent = new KeyboardEvent('keydown', { key: 'a' });
      window.dispatchEvent(aEvent);
      await editor.updateComplete;
      
      expect(editor.cursorX).toBe(2);
      expect(editor.content[0]).toBe('test');
      
      const hiddenInput = editor.shadowRoot?.querySelector('input');
      if (hiddenInput) {
        hiddenInput.value = 'x';
        const inputEvent = new Event('input', { bubbles: true });
        hiddenInput.dispatchEvent(inputEvent);
        await editor.updateComplete;
        
        expect(editor.content[0]).toBe('texst');
      }
    });
  });

  describe('i command (insert before cursor)', () => {
    it('should enter insert mode without moving cursor', async () => {
      editor.setContent(['hello world']);
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const event = new KeyboardEvent('keydown', { key: 'i' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
      const status = editor.getStatus();
      expect(status.mode).toBe('insert');
      expect(status.cursorX).toBe(5);
      expect(status.cursorY).toBe(0);
    });

    it('should NOT insert "i" character when entering insert mode', async () => {
      editor.setContent(['hello world']);
      editor.cursorX = 3;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const event = new KeyboardEvent('keydown', { key: 'i' });
      window.dispatchEvent(event);
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('hello world');
      expect(editor.cursorX).toBe(3);
      const status = editor.getStatus();
      expect(status.mode).toBe('insert');
    });

    it('should allow typing after pressing i', async () => {
      editor.setContent(['test']);
      editor.cursorX = 2;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const iEvent = new KeyboardEvent('keydown', { key: 'i' });
      window.dispatchEvent(iEvent);
      await editor.updateComplete;
      
      expect(editor.cursorX).toBe(2);
      expect(editor.content[0]).toBe('test');
      
      const hiddenInput = editor.shadowRoot?.querySelector('input');
      if (hiddenInput) {
        hiddenInput.value = 'x';
        const inputEvent = new Event('input', { bubbles: true });
        hiddenInput.dispatchEvent(inputEvent);
        await editor.updateComplete;
        
        expect(editor.content[0]).toBe('texst');
      }
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
      expect(status.mode).toBe('fast-match');
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
      
      expect(editor.getStatus().mode).toBe('fast-match');
      
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
      
      expect(editor.getStatus().mode).toBe('fast-match');
      
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

  describe('vi` command (visual select backtick)', () => {
    it('should select content between backticks', async () => {
      editor.setContent(['const str = `hello world`;']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', '`');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('visual');
      expect(editor.visualStartX).toBe(13);
      expect(editor.visualStartY).toBe(0);
      expect(editor.cursorX).toBe(23);
      expect(editor.cursorY).toBe(0);
    });

    it('should select and delete with vi`x', async () => {
      editor.setContent(['const str = `hello world`;']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', '`', 'x');
      
      expect(editor.content[0]).toBe('const str = ``;');
      expect(editor.mode).toBe('normal');
    });

    it('should handle cursor on opening backtick', async () => {
      editor.setContent(['`test content`']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', '`');
      
      expect(editor.mode).toBe('visual');
      expect(editor.visualStartX).toBe(1);
      expect(editor.cursorX).toBe(12);
    });

    it('should do nothing if no matching backticks', async () => {
      editor.setContent(['no backticks here']);
      await editor.updateComplete;
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      const originalX = editor.cursorX;
      pressKeys('v', 'i', '`');
      
      expect(editor.mode).toBe('visual');
      expect(editor.cursorX).toBe(originalX);
    });
  });

  describe("vi' command (visual select single quote)", () => {
    it('should select content between single quotes', async () => {
      editor.setContent(["const str = 'hello world';"]);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', "'");
      
      const status = editor.getStatus();
      expect(status.mode).toBe('visual');
      expect(editor.visualStartX).toBe(13);
      expect(editor.cursorX).toBe(23);
    });

    it('should handle escaped single quote', async () => {
      editor.setContent(["const str = 'don\\'t worry';"]);
      await editor.updateComplete;
      editor.cursorX = 18;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', "'");
      
      expect(editor.mode).toBe('visual');
      expect(editor.visualStartX).toBe(13);
      expect(editor.cursorX).toBe(24);
    });

    it('should select and yank with vi\'y', async () => {
      editor.setContent(["const str = 'hello world';"]);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      const mockWriteText = vi.fn();
      Object.defineProperty(navigator, 'clipboard', {
        value: { writeText: mockWriteText },
        writable: true,
        configurable: true
      });
      
      pressKeys('v', 'i', "'", 'y');
      
      expect(mockWriteText).toHaveBeenCalledWith('hello world');
      expect(editor.mode).toBe('normal');
    });
  });

  describe('vi" command (visual select double quote)', () => {
    it('should select content between double quotes', async () => {
      editor.setContent(['const str = "hello world";']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', '"');
      
      const status = editor.getStatus();
      expect(status.mode).toBe('visual');
      expect(editor.visualStartX).toBe(13);
      expect(editor.cursorX).toBe(23);
    });

    it('should handle escaped double quote', async () => {
      editor.setContent(['const str = "say \\"hello\\"";']);
      await editor.updateComplete;
      editor.cursorX = 20;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', '"');
      
      expect(editor.mode).toBe('visual');
      expect(editor.visualStartX).toBe(13);
      expect(editor.cursorX).toBe(25);
    });

    it('should select and delete with vi"d', async () => {
      editor.setContent(['const str = "hello world";']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', '"', 'd');
      
      expect(editor.content[0]).toBe('const str = "";');
      expect(editor.mode).toBe('normal');
    });

    it('should handle cursor on closing quote', async () => {
      editor.setContent(['"test content"']);
      await editor.updateComplete;
      editor.cursorX = 13;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', '"');
      
      expect(editor.mode).toBe('visual');
      expect(editor.visualStartX).toBe(1);
      expect(editor.cursorX).toBe(12);
    });
  });

  describe('viw command', () => {
    it('should select inner word from normal mode', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', 'w');
      
      expect(editor.mode).toBe('visual');
      expect(editor.visualStartX).toBe(0);
      expect(editor.visualStartY).toBe(0);
      expect(editor.cursorX).toBe(4);
      expect(editor.cursorY).toBe(0);
    });

    it('should select word in the middle', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 7;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', 'w');
      
      expect(editor.mode).toBe('visual');
      expect(editor.visualStartX).toBe(6);
      expect(editor.cursorX).toBe(10);
    });

    it('should select Chinese word', async () => {
      editor.setContent(['你好 世界 測試']);
      await editor.updateComplete;
      editor.cursorX = 3;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', 'w');
      
      expect(editor.mode).toBe('visual');
      expect(editor.visualStartX).toBe(3);
      expect(editor.cursorX).toBe(4);
    });

    it('should not select when cursor is on space', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', 'w');
      
      expect(editor.mode).toBe('normal');
    });

    it('should select word with cursor at end of word', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 4;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', 'w');
      
      expect(editor.mode).toBe('visual');
      expect(editor.visualStartX).toBe(0);
      expect(editor.cursorX).toBe(4);
    });

    it('should select and delete word with viwx', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 7;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', 'w', 'x');
      
      expect(editor.content).toEqual(['hello  test']);
      expect(editor.mode).toBe('normal');
    });

    it('should select word with numbers', async () => {
      editor.setContent(['test123 world']);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', 'w');
      
      expect(editor.mode).toBe('visual');
      expect(editor.visualStartX).toBe(0);
      expect(editor.cursorX).toBe(6);
    });
  });

  describe('multiline quote support', () => {
    it('should delete multiline content with di` (backtick)', async () => {
      editor.setContent([
        'const str = `line1',
        'line2',
        'line3`;'
      ]);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 1;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', '`');
      
      expect(editor.content).toEqual(['const str = ``;']);
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(13);
    });

    it("should delete multiline content with di' (single quote)", async () => {
      editor.setContent([
        "const str = 'line1",
        "line2",
        "line3';"
      ]);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 1;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', "'");
      
      expect(editor.content).toEqual(["const str = '';"]);
      expect(editor.cursorY).toBe(0);
    });

    it('should delete multiline content with di" (double quote)', async () => {
      editor.setContent([
        'const str = "line1',
        'line2',
        'line3";'
      ]);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 1;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', '"');
      
      expect(editor.content).toEqual(['const str = "";']);
      expect(editor.cursorY).toBe(0);
    });

    it('should select multiline content with vi` (backtick)', async () => {
      editor.setContent([
        'const str = `line1',
        'line2',
        'line3`;'
      ]);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 1;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', '`');
      
      expect(editor.mode).toBe('visual');
      expect(editor.visualStartY).toBe(0);
      expect(editor.visualStartX).toBe(13);
      expect(editor.cursorY).toBe(2);
      expect(editor.cursorX).toBe(4);
    });

    it("should select multiline content with vi' (single quote)", async () => {
      editor.setContent([
        "const str = 'line1",
        "line2",
        "line3';"
      ]);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 1;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', "'");
      
      expect(editor.mode).toBe('visual');
      expect(editor.visualStartY).toBe(0);
      expect(editor.visualStartX).toBe(13);
      expect(editor.cursorY).toBe(2);
      expect(editor.cursorX).toBe(4);
    });

    it('should select multiline content with vi" (double quote)', async () => {
      editor.setContent([
        'const str = "line1',
        'line2',
        'line3";'
      ]);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 1;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', '"');
      
      expect(editor.mode).toBe('visual');
      expect(editor.visualStartY).toBe(0);
      expect(editor.visualStartX).toBe(13);
      expect(editor.cursorY).toBe(2);
      expect(editor.cursorX).toBe(4);
    });

    it('should handle cursor on first line of multiline quote', async () => {
      editor.setContent([
        'const str = `hello',
        'world`;'
      ]);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', '`');
      
      expect(editor.content).toEqual(['const str = ``;']);
    });

    it('should handle cursor on last line of multiline quote', async () => {
      editor.setContent([
        'const str = `hello',
        'world`;'
      ]);
      await editor.updateComplete;
      editor.cursorX = 3;
      editor.cursorY = 1;
      editor.mode = 'normal';
      
      pressKeys('d', 'i', '`');
      
      expect(editor.content).toEqual(['const str = ``;']);
    });

    it('should select multiline with vi` before deleting', async () => {
      editor.setContent([
        'const str = `hello',
        'world`;'
      ]);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', '`');
      
      expect(editor.mode).toBe('visual');
      expect(editor.visualStartY).toBe(0);
      expect(editor.visualStartX).toBe(13);
      expect(editor.cursorY).toBe(1);
      expect(editor.cursorX).toBe(4);
    });

    it('should select and delete multiline with vi`x', async () => {
      editor.setContent([
        'const str = `hello',
        'world`;'
      ]);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('v', 'i', '`', 'x');
      
      expect(editor.content).toEqual(['const str = ``;']);
      expect(editor.mode).toBe('normal');
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

  describe('% command (jump to matching bracket)', () => {
    it('should jump from opening bracket to closing bracket', async () => {
      editor.setContent(['function test() {', '  return true;', '}']);
      await editor.updateComplete;
      editor.cursorX = 16;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(2);
      expect(editor.cursorX).toBe(0);
    });

    it('should jump from closing bracket to opening bracket', async () => {
      editor.setContent(['function test() {', '  return true;', '}']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 2;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(16);
    });

    it('should jump from opening parenthesis to closing parenthesis', async () => {
      editor.setContent(['const result = (1 + 2) * 3;']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(21);
    });

    it('should jump from closing parenthesis to opening parenthesis', async () => {
      editor.setContent(['const result = (1 + 2) * 3;']);
      await editor.updateComplete;
      editor.cursorX = 21;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(15);
    });

    it('should handle nested brackets', async () => {
      editor.setContent(['const arr = [[1, 2], [3, 4]];']);
      await editor.updateComplete;
      editor.cursorX = 12;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(27);
    });

    it('should handle nested brackets - inner bracket', async () => {
      editor.setContent(['const arr = [[1, 2], [3, 4]];']);
      await editor.updateComplete;
      editor.cursorX = 13;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(18);
    });

    it('should jump from opening quote to closing quote', async () => {
      editor.setContent(['const str = "hello world";']);
      await editor.updateComplete;
      editor.cursorX = 12;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(24);
    });

    it('should jump from closing quote to opening quote', async () => {
      editor.setContent(['const str = "hello world";']);
      await editor.updateComplete;
      editor.cursorX = 24;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(24);
    });

    it('should handle single quotes', async () => {
      editor.setContent(["const str = 'hello world';"]);
      await editor.updateComplete;
      editor.cursorX = 12;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(24);
    });

    it('should handle backticks', async () => {
      editor.setContent(['const str = `hello world`;']);
      await editor.updateComplete;
      editor.cursorX = 12;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(24);
    });

    it('should handle angle brackets', async () => {
      editor.setContent(['const html = <div>content</div>;']);
      await editor.updateComplete;
      editor.cursorX = 13;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(17);
    });

    it('should handle escaped quotes', async () => {
      editor.setContent(['const str = "say \\"hello\\"";']);
      await editor.updateComplete;
      editor.cursorX = 12;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(26);
    });

    it('should handle multiline brackets', async () => {
      editor.setContent([
        'function test() {',
        '  if (true) {',
        '    return 1;',
        '  }',
        '}'
      ]);
      await editor.updateComplete;
      editor.cursorX = 16;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(4);
      expect(editor.cursorX).toBe(0);
    });

    it('should handle multiline quotes', async () => {
      editor.setContent([
        'const str = "line1',
        'line2',
        'line3";'
      ]);
      await editor.updateComplete;
      editor.cursorX = 12;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(2);
      expect(editor.cursorX).toBe(5);
    });

    it('should not move cursor if no matching bracket found', async () => {
      editor.setContent(['const x = 5;']);
      await editor.updateComplete;
      editor.cursorX = 6;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(6);
    });

    it('should not move cursor if not on a bracket', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorX = 3;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(3);
    });

    it('should handle complex nested structure', async () => {
      editor.setContent(['const obj = { a: [1, (2 + 3), 4], b: "test" };']);
      await editor.updateComplete;
      editor.cursorX = 12;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(44);
    });

    it('should handle nested parentheses in array', async () => {
      editor.setContent(['const arr = [(1 + 2), (3 + 4)];']);
      await editor.updateComplete;
      editor.cursorX = 13;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(19);
    });

    it('should jump backward from inner closing bracket', async () => {
      editor.setContent(['const arr = [[1, 2], [3, 4]];']);
      await editor.updateComplete;
      editor.cursorX = 18;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(13);
    });

    it('should handle escaped backslash before quote', async () => {
      editor.setContent(['const str = "test\\\\\\" string";']);
      await editor.updateComplete;
      editor.cursorX = 12;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('%');
      
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(28);
    });
  });

  describe('di% command (delete inner bracket)', () => {
    it('should delete content between brackets when cursor is inside', async () => {
      editor.setContent(['const arr = [1, 2, 3];']);
      await editor.updateComplete;
      editor.cursorX = 14;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('i');
      pressKey('%');
      
      expect(editor.content[0]).toBe('const arr = [];');
      expect(editor.cursorX).toBe(13);
    });

    it('should delete content between parentheses when cursor is inside', async () => {
      editor.setContent(['function test(a, b, c) {']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('i');
      pressKey('%');
      
      expect(editor.content[0]).toBe('function test() {');
      expect(editor.cursorX).toBe(14);
    });

    it('should delete content between curly braces when cursor is inside', async () => {
      editor.setContent(['if (true) { console.log("test"); }']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('i');
      pressKey('%');
      
      expect(editor.content[0]).toBe('if (true) {}');
      expect(editor.cursorX).toBe(11);
    });

    it('should delete content between double quotes when cursor is inside', async () => {
      editor.setContent(['const str = "hello world";']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('i');
      pressKey('%');
      
      expect(editor.content[0]).toBe('const str = "";');
      expect(editor.cursorX).toBe(13);
    });

    it('should delete content between single quotes when cursor is inside', async () => {
      editor.setContent(["const str = 'hello world';"]);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('i');
      pressKey('%');
      
      expect(editor.content[0]).toBe("const str = '';");
      expect(editor.cursorX).toBe(13);
    });

    it('should delete content between backticks when cursor is inside', async () => {
      editor.setContent(['const str = `hello ${name}`;']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('i');
      pressKey('%');
      
      expect(editor.content[0]).toBe('const str = ``;');
      expect(editor.cursorX).toBe(13);
    });

    it('should delete content between angle brackets when cursor is inside', async () => {
      editor.setContent(['const tag = <div>content</div>;']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('i');
      pressKey('%');
      
      expect(editor.content[0]).toBe('const tag = <>content</div>;');
      expect(editor.cursorX).toBe(13);
    });

    it('should handle nested brackets correctly', async () => {
      editor.setContent(['const arr = [[1, 2], [3, 4]];']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('i');
      pressKey('%');
      
      expect(editor.content[0]).toBe('const arr = [[], [3, 4]];');
      expect(editor.cursorX).toBe(14);
    });

    it('should handle deeply nested structures', async () => {
      editor.setContent(['const obj = { a: { b: { c: 1 } } };']);
      await editor.updateComplete;
      editor.cursorX = 20;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('i');
      pressKey('%');
      
      expect(editor.content[0]).toBe('const obj = { a: {} };');
      expect(editor.cursorX).toBe(18);
    });

    it('should handle multiline brackets', async () => {
      editor.setContent([
        'function test() {',
        '  const x = 1;',
        '  return x;',
        '}'
      ]);
      await editor.updateComplete;
      editor.cursorX = 5;
      editor.cursorY = 1;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('i');
      pressKey('%');
      
      expect(editor.content).toEqual(['function test() {}']);
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(17);
    });

    it('should handle multiline quotes', async () => {
      editor.setContent([
        'const str = `',
        '  hello',
        '  world',
        '`;'
      ]);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 1;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('i');
      pressKey('%');
      
      expect(editor.content).toEqual(['const str = ``;']);
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(13);
    });

    it('should do nothing if cursor is not inside any brackets', async () => {
      editor.setContent(['const x = 1;']);
      await editor.updateComplete;
      editor.cursorX = 6;
      editor.cursorY = 0;
      editor.mode = 'normal';
      const originalContent = editor.content[0];
      
      pressKey('d');
      pressKey('i');
      pressKey('%');
      
      expect(editor.content[0]).toBe(originalContent);
      expect(editor.cursorX).toBe(6);
    });

    it('should handle escaped quotes correctly', async () => {
      editor.setContent(['const str = "hello \\" world";']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('i');
      pressKey('%');
      
      expect(editor.content[0]).toBe('const str = "";');
      expect(editor.cursorX).toBe(13);
    });

    it('should find closest bracket pair when multiple options exist', async () => {
      editor.setContent(['const x = (a + (b * c)) + d;']);
      await editor.updateComplete;
      editor.cursorX = 18;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('i');
      pressKey('%');
      
      expect(editor.content[0]).toBe('const x = (a + ()) + d;');
      expect(editor.cursorX).toBe(16);
    });

    it('should support undo for di%', async () => {
      editor.setContent(['const arr = [1, 2, 3];']);
      await editor.updateComplete;
      editor.cursorX = 14;
      editor.cursorY = 0;
      editor.mode = 'normal';
      const originalContent = editor.content[0];
      
      pressKey('d');
      pressKey('i');
      pressKey('%');
      
      expect(editor.content[0]).toBe('const arr = [];');
      
      pressKey('u');
      
      expect(editor.content[0]).toBe(originalContent);
    });

    it('should handle cursor at bracket position', async () => {
      editor.setContent(['const arr = [1, 2, 3];']);
      await editor.updateComplete;
      editor.cursorX = 12;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('i');
      pressKey('%');
      
      expect(editor.content[0]).toBe('const arr = [];');
      expect(editor.cursorX).toBe(13);
    });
  });

  describe('da( command (delete around parentheses)', () => {
    it('should delete content including parentheses when cursor is inside', async () => {
      editor.setContent(['function test(a, b, c) {']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('(');
      
      expect(editor.content[0]).toBe('function test {');
      expect(editor.cursorX).toBe(13);
    });

    it('should delete content including parentheses when using closing bracket', async () => {
      editor.setContent(['function test(a, b, c) {']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey(')');
      
      expect(editor.content[0]).toBe('function test {');
      expect(editor.cursorX).toBe(13);
    });

    it('should handle nested parentheses', async () => {
      editor.setContent(['const x = (a + (b * c)) + d;']);
      await editor.updateComplete;
      editor.cursorX = 18;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('(');
      
      expect(editor.content[0]).toBe('const x = (a + ) + d;');
      expect(editor.cursorX).toBe(15);
    });

    it('should handle empty parentheses', async () => {
      editor.setContent(['function test() {']);
      await editor.updateComplete;
      editor.cursorX = 14;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('(');
      
      expect(editor.content[0]).toBe('function test {');
      expect(editor.cursorX).toBe(13);
    });
  });

  describe('da[ command (delete around square brackets)', () => {
    it('should delete content including brackets when cursor is inside', async () => {
      editor.setContent(['const arr = [1, 2, 3];']);
      await editor.updateComplete;
      editor.cursorX = 14;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('[');
      
      expect(editor.content[0]).toBe('const arr = ;');
      expect(editor.cursorX).toBe(12);
    });

    it('should work with closing bracket', async () => {
      editor.setContent(['const arr = [1, 2, 3];']);
      await editor.updateComplete;
      editor.cursorX = 14;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey(']');
      
      expect(editor.content[0]).toBe('const arr = ;');
      expect(editor.cursorX).toBe(12);
    });

    it('should handle nested brackets', async () => {
      editor.setContent(['const arr = [[1, 2], [3, 4]];']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('[');
      
      expect(editor.content[0]).toBe('const arr = [, [3, 4]];');
      expect(editor.cursorX).toBe(13);
    });
  });

  describe('da{ command (delete around curly braces)', () => {
    it('should delete content including braces when cursor is inside', async () => {
      editor.setContent(['if (true) { console.log("test"); }']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('{');
      
      expect(editor.content[0]).toBe('if (true) ');
      expect(editor.cursorX).toBe(9);
    });

    it('should work with closing brace', async () => {
      editor.setContent(['if (true) { console.log("test"); }']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('}');
      
      expect(editor.content[0]).toBe('if (true) ');
      expect(editor.cursorX).toBe(9);
    });

    it('should handle multiline braces', async () => {
      editor.setContent([
        'function test() {',
        '  const x = 1;',
        '  return x;',
        '}'
      ]);
      await editor.updateComplete;
      editor.cursorX = 5;
      editor.cursorY = 1;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('{');
      
      expect(editor.content).toEqual(['function test() ']);
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(15);
    });
  });

  describe('da< command (delete around angle brackets)', () => {
    it('should delete content including angle brackets when cursor is inside', async () => {
      editor.setContent(['const tag = <div>content</div>;']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('<');
      
      expect(editor.content[0]).toBe('const tag = content</div>;');
      expect(editor.cursorX).toBe(12);
    });

    it('should work with closing angle bracket', async () => {
      editor.setContent(['const tag = <div>content</div>;']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('>');
      
      expect(editor.content[0]).toBe('const tag = content</div>;');
      expect(editor.cursorX).toBe(12);
    });
  });

  describe('da" command (delete around double quotes)', () => {
    it('should delete content including quotes when cursor is inside', async () => {
      editor.setContent(['const str = "hello world";']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('"');
      
      expect(editor.content[0]).toBe('const str = ;');
      expect(editor.cursorX).toBe(12);
    });

    it('should handle empty quotes', async () => {
      editor.setContent(['const str = "";']);
      await editor.updateComplete;
      editor.cursorX = 13;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('"');
      
      expect(editor.content[0]).toBe('const str = ;');
      expect(editor.cursorX).toBe(12);
    });

    it('should handle multiline quotes', async () => {
      editor.setContent([
        'const str = "',
        '  hello',
        '  world',
        '";'
      ]);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 1;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('"');
      
      expect(editor.content).toEqual(['const str = ;']);
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(12);
    });
  });

  describe("da' command (delete around single quotes)", () => {
    it('should delete content including quotes when cursor is inside', async () => {
      editor.setContent(["const str = 'hello world';"]);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey("'");
      
      expect(editor.content[0]).toBe('const str = ;');
      expect(editor.cursorX).toBe(12);
    });

    it('should handle escaped quotes', async () => {
      editor.setContent(["const str = 'hello \\' world';"]);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey("'");
      
      expect(editor.content[0]).toBe('const str = ;');
      expect(editor.cursorX).toBe(12);
    });
  });

  describe('da` command (delete around backticks)', () => {
    it('should delete content including backticks when cursor is inside', async () => {
      editor.setContent(['const str = `hello ${name}`;']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('`');
      
      expect(editor.content[0]).toBe('const str = ;');
      expect(editor.cursorX).toBe(12);
    });

    it('should handle multiline template literals', async () => {
      editor.setContent([
        'const str = `',
        '  hello',
        '  world',
        '`;'
      ]);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 1;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('`');
      
      expect(editor.content).toEqual(['const str = ;']);
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(12);
    });
  });

  describe('da% command (delete around any bracket)', () => {
    it('should delete content including brackets when cursor is inside parentheses', async () => {
      editor.setContent(['function test(a, b, c) {']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('%');
      
      expect(editor.content[0]).toBe('function test {');
      expect(editor.cursorX).toBe(13);
    });

    it('should delete content including brackets when cursor is inside square brackets', async () => {
      editor.setContent(['const arr = [1, 2, 3];']);
      await editor.updateComplete;
      editor.cursorX = 14;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('%');
      
      expect(editor.content[0]).toBe('const arr = ;');
      expect(editor.cursorX).toBe(12);
    });

    it('should delete content including brackets when cursor is inside curly braces', async () => {
      editor.setContent(['if (true) { console.log("test"); }']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('%');
      
      expect(editor.content[0]).toBe('if (true) ');
      expect(editor.cursorX).toBe(9);
    });

    it('should delete content including quotes when cursor is inside quotes', async () => {
      editor.setContent(['const str = "hello world";']);
      await editor.updateComplete;
      editor.cursorX = 15;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      pressKey('a');
      pressKey('%');
      
      expect(editor.content[0]).toBe('const str = ;');
      expect(editor.cursorX).toBe(12);
    });

    it('should support undo for da%', async () => {
      editor.setContent(['const arr = [1, 2, 3];']);
      await editor.updateComplete;
      editor.cursorX = 14;
      editor.cursorY = 0;
      editor.mode = 'normal';
      const originalContent = editor.content[0];
      
      pressKey('d');
      pressKey('a');
      pressKey('%');
      
      expect(editor.content[0]).toBe('const arr = ;');
      
      pressKey('u');
      
      expect(editor.content[0]).toBe(originalContent);
    });
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

