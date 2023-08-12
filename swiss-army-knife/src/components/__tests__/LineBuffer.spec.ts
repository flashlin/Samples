import { beforeEach, describe, expect, it, vi, type MockedObject } from 'vitest';
import { LineBuffer, EditorBuffer } from '../EditorBuffer';

describe('LineBuffer', () => {
  const mockEditorBuffer = EditorBuffer as MockedObject<typeof EditorBuffer>

  beforeEach(() => {
  });

  it('append abc', () => {
    const line = new LineBuffer(new mockEditorBuffer(), 0, 0);
    line.append('abc');
    expect(line.content).toBe('abc');
  });

  it('append abc...123', () => {
    const line = new LineBuffer(new mockEditorBuffer(), 0, 0);
    line.append('abc');
    line.append('123');
    expect(line.content).toBe('abc123');
  });

  it('append abc\\n123', () => {
    const line = new LineBuffer(new mockEditorBuffer(), 0, 0);
    line.append('abc\n123');
    expect(line.content).toBe('abc');
  });

  it('replace a', () => {
    const line = new LineBuffer(new mockEditorBuffer(), 0, 0);
    line.append('a');
    const replacedLength = line.replace(0, '123');
    expect(line.content).toBe('1');
    expect(replacedLength).toBe(1);
  });
});
