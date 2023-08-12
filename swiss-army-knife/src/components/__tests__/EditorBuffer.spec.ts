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
});
