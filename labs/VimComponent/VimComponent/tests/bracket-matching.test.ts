import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setupP5Mock, createTestEditor, cleanupTestEditor, pressKey, pressKeys } from './test-helpers';

setupP5Mock();

import '../src/vim-editor';

describe('VimEditor - Bracket Matching', () => {
  let editor: any;

  beforeEach(async () => {
    editor = await createTestEditor();
  });

  afterEach(() => {
    cleanupTestEditor(editor);
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

});
