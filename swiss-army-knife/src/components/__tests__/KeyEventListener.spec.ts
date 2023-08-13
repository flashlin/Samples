import { beforeEach, describe, expect, it, vi, test } from 'vitest';
import { Subject } from 'rxjs';
import { NumMoveListener, type IEditor } from '../EditorBuffer';

describe('KeyEvent', () => {
  beforeEach(() => { });

  it('event 123a', () => {
    const keyboardEvent = new Subject<KeyboardEvent>();
    const mockCallback = vi.fn();//.mockImplementation(() => (flag = signal));
    const sut = new NumMoveListener();
    sut.listenEvent({} as IEditor, keyboardEvent);
    sut.attach(mockCallback);
    emitKeys(keyboardEvent, '12a');
    emitKeys(keyboardEvent, '123b');
    expect(mockCallback).toHaveBeenCalledTimes(2);
  });

  test.each([
    ["123a", 1],
    ["a", 0],
    ["a12", 0],
  ])("input keyEvent '%s'", (input, expectedCalledTimes) => {
    const keyboardEvent = new Subject<KeyboardEvent>();
    const mockCallback = vi.fn();//.mockImplementation(() => (flag = signal));
    const sut = new NumMoveListener();
    sut.listenEvent({} as IEditor, keyboardEvent);
    sut.attach(mockCallback);
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

