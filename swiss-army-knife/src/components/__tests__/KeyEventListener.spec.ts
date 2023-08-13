import { beforeEach, describe, expect, it, vi, test } from 'vitest';
import { Subject } from 'rxjs';
import { NumMoveListener } from '../EditorBuffer';

describe('KeyEvent', () => {
  beforeEach(() => { });

  it('event 123%', () => {
    const keyboardEvent = new Subject<KeyboardEvent>();
    const mockCallback = vi.fn();//.mockImplementation(() => (flag = signal));
    const sut = new NumMoveListener(keyboardEvent);
    sut.listen(mockCallback);
    emitKeys(keyboardEvent, '123%');
    expect(mockCallback).toHaveBeenCalledTimes(0);
  });

  test.each([
    ["123a", 1],
    ["a", 0],
    ["a12", 0],
  ])("input keyEvent '%s'", (input, expectedCalledTimes) => {
    const keyboardEvent = new Subject<KeyboardEvent>();
    const mockCallback = vi.fn();//.mockImplementation(() => (flag = signal));
    const sut = new NumMoveListener(keyboardEvent);
    sut.listen(mockCallback);
    emitKeys(keyboardEvent, input);
    expect(mockCallback).toHaveBeenCalledTimes(expectedCalledTimes);
  })
});



function emitKey(keyboardEvent: Subject<KeyboardEvent>, key: string) {
  keyboardEvent.next(new KeyboardEvent('keydown', { key: key }));
}

function emitKeys(keyboardEvent: Subject<KeyboardEvent>, keys: string) {
  for (const key of keys) {
    emitKey(keyboardEvent, key);
  }
}
function testEach(myTest: (input: string, expected: string) => void, arg1: string[], arg2: string[]) {
  throw new Error('Function not implemented.');
}

