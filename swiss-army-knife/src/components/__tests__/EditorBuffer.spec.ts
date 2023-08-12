import { beforeEach, describe, expect, it } from 'vitest';
import { EditorBuffer } from '../EditorBuffer';

describe('EditorBuffer', () => {
  beforeEach(() => {
  });

  it('append abc', () => {
    const editor = new EditorBuffer();
    editor.appendLine(0, 0, "abc")
    expect(editor.getContent()).toBe('abc');
  });

  it('append abc\n123', () => {
    const editor = new EditorBuffer();
    editor.appendLine(0, 0, "abc\n123")
    expect(editor.getContent()).toBe('abc\n123');
    expect(editor.lines.length).toBe(2);
  });

  it('append abc\n123', () => {
    const editor = new EditorBuffer();
    editor.appendLine(0, 0, "abc\n123")
    editor.appendLine(0, 1, "ABC")
    expect(editor.getContent()).toBe('aABCbc\n123');
    expect(editor.lines.length).toBe(2);
  });
});
