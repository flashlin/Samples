import { beforeEach, describe, expect, it, vi } from 'vitest';
import { LineBuffer, EditorBuffer } from '../LineBuffer';

vi.mock('../LineBuffer', async () => {
  const actual = await vi.importActual("../LineBuffer") as object
  return {
    ...actual,
    EditorBuffer: vi.fn(),
  };
});

describe('LineBuffer', () => {
  const mockEditorBuffer = new EditorBuffer();

  beforeEach(() => {
  });

  it('append abc', () => {
    const line = new LineBuffer(mockEditorBuffer, 0, 0);
    line.append('abc');
    expect(line.content).toBe('abc');
  });

  it('append 123', () => {
    const line = new LineBuffer(mockEditorBuffer, 0, 0);
    line.append('abc');
    line.append('123');
    expect(line.content).toBe('abc123');
  });
});
