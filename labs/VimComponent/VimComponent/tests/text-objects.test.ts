import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setupP5Mock, createTestEditor, cleanupTestEditor, pressKey, pressKeys } from './test-helpers';

setupP5Mock();

import '../src/vim-editor';

describe('VimEditor - Text Objects', () => {
  let editor: any;

  beforeEach(async () => {
    editor = await createTestEditor();
  });

  afterEach(() => {
    cleanupTestEditor(editor);
  });

  describe('diw command', () => {
    it('should delete English word under cursor', async () => {
      editor.setContent(['hello world test']);
      await editor.updateComplete;
      editor.cursorX = 7;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      
      pressKey('i');
      
      pressKey('w');
      
      expect(editor.content[0]).toBe('hello  test');
      expect(editor.cursorX).toBe(6);
    });

    it('should delete Chinese word under cursor', async () => {
      editor.setContent(['你好世界測試']);
      await editor.updateComplete;
      editor.cursorX = 2;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      
      pressKey('i');
      
      pressKey('w');
      
      expect(editor.content[0]).toBe('');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete word with numbers', async () => {
      editor.setContent(['hello world123 test']);
      await editor.updateComplete;
      editor.cursorX = 8;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      
      pressKey('i');
      
      pressKey('w');
      
      expect(editor.content[0]).toBe('hello  test');
      expect(editor.cursorX).toBe(6);
    });

    it('should delete word at start of line', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      
      pressKey('i');
      
      pressKey('w');
      
      expect(editor.content[0]).toBe(' world');
      expect(editor.cursorX).toBe(0);
    });

    it('should delete word at end of line', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorX = 10;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      
      pressKey('i');
      
      pressKey('w');
      
      expect(editor.content[0]).toBe('hello ');
      expect(editor.cursorX).toBe(5);
    });

    it('should not delete when cursor is on space', async () => {
      editor.setContent(['hello world']);
      await editor.updateComplete;
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      
      pressKey('i');
      
      pressKey('w');
      
      expect(editor.content[0]).toBe('hello world');
    });

    it('should handle mixed English and Chinese', async () => {
      editor.setContent(['hello你好world']);
      await editor.updateComplete;
      editor.cursorX = 6;
      editor.cursorY = 0;
      editor.mode = 'normal';
      
      pressKey('d');
      
      pressKey('i');
      
      pressKey('w');
      
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
      
      await pressKeys('v', 'i', '`', 'x');
      await new Promise(resolve => setTimeout(resolve, 50));
      
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
      
      await pressKeys('v', 'i', "'", 'y');
      await new Promise(resolve => setTimeout(resolve, 50));
      
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
      
      await pressKeys('v', 'i', '"', 'd');
      await new Promise(resolve => setTimeout(resolve, 50));
      
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
      
      await pressKeys('v', 'i', 'w', 'x');
      await new Promise(resolve => setTimeout(resolve, 50));
      
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
      
      await pressKeys('v', 'i', '`', 'x');
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(editor.content).toEqual(['const str = ``;']);
      expect(editor.mode).toBe('normal');
    });
  });
});
