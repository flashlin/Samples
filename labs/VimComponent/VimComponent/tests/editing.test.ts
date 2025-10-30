import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setupP5Mock, createTestEditor, cleanupTestEditor, pressKey, pressKeys, setupClipboardMock } from './test-helpers';

setupP5Mock();

import '../src/vim-editor';

describe('VimEditor - Editing', () => {
  let editor: any;

  beforeEach(async () => {
    editor = await createTestEditor();
    // Clear clipboard after creating editor to avoid pollution from previous tests
    await navigator.clipboard.writeText('');
    await new Promise(resolve => setTimeout(resolve, 20));
  });

  afterEach(() => {
    cleanupTestEditor(editor);
  });

  describe('dw command', () => {
    it('should delete word from cursor position', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      
      pressKey('w');
      
      expect(editor.content[0]).toBe('world test');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete from middle of word', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      
      pressKey('w');
      
      expect(editor.content[0]).toBe('heworld test');
      expect(editor.cursorX).toBe(2);
    });

    it('should delete Chinese word from cursor', async () => {
      editor.setContent(['你好世界測試']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      
      pressKey('w');
      
      expect(editor.content[0]).toBe('');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete word including trailing spaces', async () => {
      editor.setContent(['hello  world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      
      pressKey('w');
      
      expect(editor.content[0]).toBe('world');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete spaces when cursor on space', async () => {
      editor.setContent(['hello  world']);
      await editor.updateComplete;
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      
      pressKey('w');
      
      expect(editor.content[0]).toBe('helloworld');
      expect(editor.cursorX).toBe(5);
    });

    it('should handle word with numbers', async () => {
      editor.setContent(['hello123 world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      
      pressKey('w');
      
      expect(editor.content[0]).toBe('world');
      expect(editor.cursorX).toBe(0);
    });

    it('should handle delete at end of line', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorX = 6;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      
      pressKey('w');
      
      expect(editor.content[0]).toBe('hello ');
      expect(editor.cursorX).toBe(5);
    });

    it('should handle mixed English and Chinese with dw', async () => {
      editor.setContent(['hello你好world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      
      pressKey('w');
      
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

  describe('dt{char} command (delete till character)', () => {
    it('should delete from cursor to next space', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', ' ');
      
      expect(editor.content[0]).toBe(' world test');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete from middle of word to next space', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', ' ');
      
      expect(editor.content[0]).toBe('he world test');
      expect(editor.cursorX).toBe(2);
    });

    it('should delete to end of line if no space found', async () => {
      editor.setContent(['helloworld']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', ' ');
      
      expect(editor.content[0]).toBe('');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete to end of line when not last line', async () => {
      editor.setContent(['helloworld', 'next line']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', ' ');
      
      expect(editor.content[0]).toBe('');
      expect(editor.content[1]).toBe('next line');
    });

    it('should handle cursor before space', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorX = 4;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', ' ');
      
      expect(editor.content[0]).toBe('hell world');
      expect(editor.cursorX).toBe(4);
    });

    it('should handle cursor on space', async () => {
      editor.setContent(['hello  world']);
      await editor.updateComplete;
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', ' ');
      
      expect(editor.content[0]).toBe('hello  world');
      expect(editor.cursorX).toBe(5);
    });

    it('should handle Chinese characters', async () => {
      editor.setContent(['你好 世界 測試']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', ' ');
      
      expect(editor.content[0]).toBe(' 世界 測試');
      expect(editor.cursorX).toBe(0);
    });

    it('should handle mixed English and Chinese', async () => {
      editor.setContent(['hello你好 world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', ' ');
      
      expect(editor.content[0]).toBe(' world');
      expect(editor.cursorX).toBe(0);
    });

    it('should do nothing on empty line', async () => {
      editor.setContent(['']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', ' ');
      
      expect(editor.content[0]).toBe('');
      expect(editor.cursorX).toBe(0);
    });

    it('should support undo', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', ' ');
      expect(editor.content[0]).toBe(' world test');
      
      pressKey('u');
      expect(editor.content[0]).toBe('hello world test');
    });

    it('should handle multiple spaces', async () => {
      editor.setContent(['hello    world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', ' ');
      
      expect(editor.content[0]).toBe('    world');
      expect(editor.cursorX).toBe(0);
    });

    it('should handle cursor at end of line', async () => {
      editor.setContent(['hello world', 'next']);
      await editor.updateComplete;
      editor.cursorX = 11;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', ' ');
      
      expect(editor.content[0]).toBe('hello world');
      expect(editor.content[1]).toBe('next');
    });

    it('should delete till character a', async () => {
      editor.setContent(['hello world and test']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', 'a');
      
      expect(editor.content[0]).toBe('and test');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete till comma', async () => {
      editor.setContent(['one, two, three']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', ',');
      
      expect(editor.content[0]).toBe(', two, three');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete till Chinese character', async () => {
      editor.setContent(['hello世界test']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', '世');
      
      expect(editor.content[0]).toBe('世界test');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete till dot from middle of word', async () => {
      editor.setContent(['hello.world.test']);
      await editor.updateComplete;
      editor.cursorX = 6;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', '.');
      
      expect(editor.content[0]).toBe('hello..test');
      expect(editor.cursorX).toBe(6);
    });

    it('should handle special characters', async () => {
      editor.setContent(['test@example.com']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', '@');
      
      expect(editor.content[0]).toBe('@example.com');
      expect(editor.cursorX).toBe(0);
    });

    it('should support undo for dt{char}', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKeys('d', 't', 'w');
      expect(editor.content[0]).toBe('world test');
      
      pressKey('u');
      expect(editor.content[0]).toBe('hello world test');
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

      pressKey('p');

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

      pressKey('p');

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

      pressKey('p');

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

      pressKey('p');

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

      pressKey('p');

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

      pressKey('p');

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

      pressKey('p');

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

      pressKey('u');
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

      pressKey('d');

      pressKey('i');

      pressKey('w');
      await editor.updateComplete;

      expect(editor.content[0]).toBe('hello  test');

      pressKey('u');
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

      pressKey('d');

      pressKey('w');
      await editor.updateComplete;

      expect(editor.content[0]).toBe('world');

      pressKey('u');
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

      pressKey('p');

      await new Promise(resolve => setTimeout(resolve, 10));
      await editor.updateComplete;

      expect(editor.content[0]).toBe('hello world');

      pressKey('u');
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

      pressKey('v');
      await editor.updateComplete;

      editor.cursorX = 4;

      await pressKey('d');
      await editor.updateComplete;
      await new Promise(resolve => setTimeout(resolve, 50));

      expect(editor.content[0]).toBe(' world');

      pressKey('u');
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

      pressKey('d');
      pressKey('i');
      pressKey('w');
      await editor.updateComplete;

      expect(editor.content[0]).toBe(' world');

      pressKey('u');
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

      pressKey('d');
      pressKey('2');
      pressKey('j');
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

      pressKey('d');
      pressKey('2');
      pressKey('k');
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

      pressKey('d');
      pressKey('1');
      pressKey('j');
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

      pressKey('d');
      pressKey('5');
      pressKey('j');
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

      pressKey('d');
      pressKey('5');
      pressKey('k');
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

      pressKey('d');
      pressKey('3');
      pressKey('j');
      await editor.updateComplete;

      expect(editor.content).toEqual(['line1', 'line6']);

      pressKey('u');
      await editor.updateComplete;

      expect(editor.content).toEqual(['line1', 'line2', 'line3', 'line4', 'line5', 'line6']);
      expect(editor.cursorY).toBe(1);
    });
  });

  describe('d$ command (delete to end of line)', () => {
    it('should delete from cursor to end of line', async () => {
      editor.content = ['hello world', 'test line'];
      editor.cursorY = 0;
      editor.cursorX = 6;
      
      await pressKeys('d', '$');
      
      expect(editor.content[0]).toBe('hello ');
      expect(editor.cursorX).toBe(5);
    });
    
    it('should do nothing if cursor is at end of line', async () => {
      editor.content = ['hello'];
      editor.cursorY = 0;
      editor.cursorX = 5;
      
      await pressKeys('d', '$');
      
      expect(editor.content[0]).toBe('hello');
    });
    
    it('should support undo after d$', async () => {
      editor.content = ['hello world'];
      editor.cursorY = 0;
      editor.cursorX = 6;
      
      await pressKeys('d', '$');
      expect(editor.content[0]).toBe('hello ');
      
      await pressKey('u');
      expect(editor.content[0]).toBe('hello world');
    });
    
    it('should handle Chinese characters correctly', async () => {
      editor.content = ['你好世界'];
      editor.cursorY = 0;
      editor.cursorX = 2;
      
      await pressKeys('d', '$');
      
      expect(editor.content[0]).toBe('你好');
      expect(editor.cursorX).toBe(1);
    });
  });

  describe('dd command (delete current line)', () => {
    it('should delete current line and move to beginning of next line', async () => {
      editor.content = ['line 1', 'line 2', 'line 3'];
      editor.cursorY = 1;
      editor.cursorX = 3;
      
      await pressKeys('d', 'd');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content).toEqual(['line 1', 'line 3']);
      expect(editor.cursorY).toBe(1);
      expect(editor.cursorX).toBe(0);
    });
    
    it('should delete first line', async () => {
      editor.content = ['first', 'second', 'third'];
      editor.cursorY = 0;
      editor.cursorX = 2;
      
      await pressKeys('d', 'd');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content).toEqual(['second', 'third']);
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(0);
    });
    
    it('should delete last line', async () => {
      editor.content = ['first', 'second', 'third'];
      editor.cursorY = 2;
      editor.cursorX = 1;
      
      await pressKeys('d', 'd');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content).toEqual(['first', 'second']);
      expect(editor.cursorY).toBe(1);
      expect(editor.cursorX).toBe(0);
    });
    
    it('should leave empty line when deleting only line', async () => {
      editor.content = ['only line'];
      editor.cursorY = 0;
      editor.cursorX = 3;
      
      await pressKeys('d', 'd');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content).toEqual(['']);
      expect(editor.cursorY).toBe(0);
      expect(editor.cursorX).toBe(0);
    });
    
    it('should support undo after dd', async () => {
      editor.content = ['line 1', 'line 2', 'line 3'];
      editor.cursorY = 1;
      editor.cursorX = 0;
      
      await pressKeys('d', 'd');
      await new Promise(resolve => setTimeout(resolve, 50));
      expect(editor.content).toEqual(['line 1', 'line 3']);
      
      await pressKey('u');
      expect(editor.content).toEqual(['line 1', 'line 2', 'line 3']);
    });
    
    it('should handle Chinese content', async () => {
      editor.content = ['第一行', '第二行', '第三行'];
      editor.cursorY = 1;
      editor.cursorX = 1;
      
      await pressKeys('d', 'd');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content).toEqual(['第一行', '第三行']);
      expect(editor.cursorY).toBe(1);
      expect(editor.cursorX).toBe(0);
    });
  });

  // NOTE: P command tests have been moved to clipboard.test.ts to avoid clipboard state pollution

  // NOTE: yy command tests have been moved to clipboard.test.ts to avoid clipboard state pollution

  describe('D command (delete to end of line)', () => {
    it('should delete from cursor to end of line', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 6;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKey('D');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('hello ');
      expect(editor.cursorX).toBe(5);
    });

    it('should delete entire line when cursor at beginning', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 0;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKey('D');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('');
      expect(editor.cursorX).toBe(0);
    });

    it('should do nothing if cursor is at end of line', async () => {
      editor.setContent(['hello']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 5;
      editor.mode = 'normal';
      
      await pressKey('D');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('hello');
    });

    it('should handle Chinese characters correctly', async () => {
      editor.setContent(['你好世界測試']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 2;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKey('D');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('你好');
      expect(editor.cursorX).toBe(1);
    });

    it('should handle empty line', async () => {
      editor.setContent(['']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 0;
      editor.mode = 'normal';
      
      await pressKey('D');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('');
      expect(editor.cursorX).toBe(0);
    });

    it('should support undo after D', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 6;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKey('D');
      await editor.updateComplete;
      expect(editor.content[0]).toBe('hello ');
      
      await pressKey('u');
      expect(editor.content[0]).toBe('hello world test');
      expect(editor.cursorX).toBe(6);
    });

    it('should handle cursor in middle of line', async () => {
      editor.setContent(['abcdefghijk']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 5;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKey('D');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('abcde');
      expect(editor.cursorX).toBe(4);
    });

    it('should handle multi-line content correctly', async () => {
      editor.setContent(['line one', 'line two', 'line three']);
      await editor.updateComplete;
      editor.cursorY = 1;
      editor.cursorX = 5;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKey('D');
      await editor.updateComplete;
      
      expect(editor.content).toEqual(['line one', 'line ', 'line three']);
      expect(editor.cursorY).toBe(1);
      expect(editor.cursorX).toBe(4);
    });

    it('should work with mixed English and Chinese', async () => {
      editor.setContent(['hello你好world世界']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 7;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKey('D');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('hello你好');
      expect(editor.cursorX).toBe(6);
    });
  });

  describe('x command (delete character under cursor)', () => {
    it('should delete character under cursor', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 0;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKey('x');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('ello world');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete character in middle of line', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 6;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKey('x');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('hello orld');
      expect(editor.cursorX).toBe(6);
    });

    it('should delete Chinese character', async () => {
      editor.setContent(['你好世界']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 0;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKey('x');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('好世界');
      expect(editor.cursorX).toBe(0);
    });

    it('should do nothing at end of line', async () => {
      editor.setContent(['hello']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 5;
      editor.mode = 'normal';
      
      await pressKey('x');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('hello');
      expect(editor.cursorX).toBe(5);
    });

    it('should do nothing on empty line', async () => {
      editor.setContent(['']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 0;
      editor.mode = 'normal';
      
      await pressKey('x');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('');
      expect(editor.cursorX).toBe(0);
    });

    it('should support undo', async () => {
      editor.setContent(['hello']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 0;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKey('x');
      await editor.updateComplete;
      expect(editor.content[0]).toBe('ello');
      
      await pressKey('u');
      expect(editor.content[0]).toBe('hello');
    });

    it('should adjust cursor when deleting last character', async () => {
      editor.setContent(['hello']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 4;
      editor.mode = 'normal';
      
      await pressKey('x');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('hell');
      expect(editor.cursorX).toBe(3);
    });
  });

  describe('X command (delete character before cursor)', () => {
    it('should delete character before cursor', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 5;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKey('X');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('hell world');
      expect(editor.cursorX).toBe(4);
    });

    it('should do nothing at beginning of line', async () => {
      editor.setContent(['hello']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 0;
      editor.mode = 'normal';
      
      await pressKey('X');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('hello');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete Chinese character before cursor', async () => {
      editor.setContent(['你好世界']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 2;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKey('X');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('你世界');
      expect(editor.cursorX).toBe(1);
    });

    it('should support undo', async () => {
      editor.setContent(['hello']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 3;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKey('X');
      await editor.updateComplete;
      expect(editor.content[0]).toBe('helo');
      
      await pressKey('u');
      expect(editor.content[0]).toBe('hello');
    });
  });

  describe('r command (replace character)', () => {
    it('should replace character with another character', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 0;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKeys('r', 'H');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('Hello world');
      expect(editor.cursorX).toBe(0);
    });

    it('should replace character in middle of line', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 6;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKeys('r', 'W');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('hello World');
      expect(editor.cursorX).toBe(6);
    });

    it('should replace with Chinese character', async () => {
      editor.setContent(['hello']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 0;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKeys('r', '你');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('你ello');
      expect(editor.cursorX).toBe(0);
    });

    it('should replace Chinese character with English', async () => {
      editor.setContent(['你好世界']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 0;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKeys('r', 'H');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('H好世界');
      expect(editor.cursorX).toBe(0);
    });

    it('should replace with space', async () => {
      editor.setContent(['hello']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 2;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKeys('r', ' ');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('he lo');
      expect(editor.cursorX).toBe(2);
    });

    it('should do nothing at end of line', async () => {
      editor.setContent(['hello']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 5;
      editor.mode = 'normal';
      
      await pressKeys('r', 'x');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('hello');
      expect(editor.cursorX).toBe(5);
    });

    it('should do nothing on empty line', async () => {
      editor.setContent(['']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 0;
      editor.mode = 'normal';
      
      await pressKeys('r', 'x');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('');
      expect(editor.cursorX).toBe(0);
    });

    it('should support undo', async () => {
      editor.setContent(['hello']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 0;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKeys('r', 'H');
      await editor.updateComplete;
      expect(editor.content[0]).toBe('Hello');
      
      await pressKey('u');
      expect(editor.content[0]).toBe('hello');
    });

    it('should allow multiple replacements', async () => {
      editor.setContent(['hello']);
      await editor.updateComplete;
      editor.cursorY = 0;
      editor.cursorX = 0;
      editor.mode = 'normal';
      editor.resetHistory();
      
      await pressKeys('r', 'H');
      await editor.updateComplete;
      expect(editor.content[0]).toBe('Hello');
      
      editor.cursorX = 1;
      await pressKeys('r', 'a');
      await editor.updateComplete;
      expect(editor.content[0]).toBe('Hallo');
    });
  });

  // NOTE: dd command clipboard tests have been moved to clipboard.test.ts to avoid clipboard state pollution
});
