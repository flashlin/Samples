import { beforeEach, describe, expect, it } from 'vitest';
import { EditorBuffer } from '../EditorBuffer';

describe('EditorBuffer', () => {
  beforeEach(() => { });

  it('append abc', () => {
    const editor = new EditorBuffer();
    editor.appendLine(0, 0, 'abc');
    expect(editor.getContent()).toBe('abc');
  });

  it('append abc\\n123', () => {
    const editor = new EditorBuffer();
    editor.appendLine(0, 0, 'abc\n123');
    expect(editor.getContent()).toBe('abc\n123');
    expect(editor.lines.length).toBe(2);
  });

  it('append aABCbc\\n123', () => {
    const editor = new EditorBuffer();
    editor.appendLine(0, 0, 'abc\n123');
    editor.appendLine(0, 1, 'ABC');
    expect(editor.getContent()).toBe('aABCbc\n123');
    expect(editor.lines.length).toBe(2);
  });

  it('append aABC\\nbc\\n123', () => {
    const editor = new EditorBuffer();
    editor.appendLine(0, 0, 'abc\n123');
    editor.appendLine(0, 1, 'ABC\n');
    expect(editor.getContent()).toBe('aABC\nbc\n123');
    expect(editor.lines.length).toBe(3);
  });

  it('delete aABC\\nbc\\n123', () => {
    const editor = new EditorBuffer();
    editor.appendLine(0, 0, 'abc\n123');
    editor.appendLine(0, 1, 'ABC\n');
    editor.delete(0, 2, 3);
    expect(editor.getContent()).toBe('aAbc\n123');
    expect(editor.lines.length).toBe(2);
  });


  it('delete a\\nb\\nc\\n1\\n', () => {
    const editor = new EditorBuffer();
    editor.appendLine(0, 0, 'a\nb\nc\n1\n');
    editor.delete(1, 1, 3);
    expect(editor.getContent()).toBe('a\nb1\n');
    expect(editor.lines.length).toBe(2);
  });

  it('replace a\\nb\\nc\\n1\\n', () => {
    const editor = new EditorBuffer();
    editor.appendLine(0, 0, 'a\nb\nc\n1\n');
    editor.replace(1, 1, "ABC");
    expect(editor.getContent()).toBe('a\nbABC1\n');
    expect(editor.lines.length).toBe(2);
  });


  it('get a\\nb\\nc', () => {
    const editor = new EditorBuffer();
    editor.appendLine(0, 0, 'a\nb\nc');
    const fragment = editor.getFragment(1, 1, 10);
    expect(fragment).toBe('\nc');
  });
});
