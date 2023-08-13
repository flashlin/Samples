import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Subject } from 'rxjs';
import { NumMoveListener } from '../EditorBuffer';

describe('KeyEvent', () => {
  beforeEach(() => { });

  it('event 123%', () => {
    const keyboardEvent = new Subject<KeyboardEvent>();

    let flag = false;
    const mockCallback = vi.fn().mockImplementation((signal) => (flag = signal));
    const sut = new NumMoveListener(keyboardEvent);
    sut.listen(mockCallback);

    emitKeys(keyboardEvent, '123%');

    expect(mockCallback).toHaveBeenCalledTimes(1);
    expect(flag).toEqual(false);
  });
});

function emitKey(keyboardEvent: Subject<KeyboardEvent>, key: string) {
  keyboardEvent.next(new KeyboardEvent('keydown', { key: key }));
}

function emitKeys(keyboardEvent: Subject<KeyboardEvent>, keys: string) {
  for (const key of keys) {
    emitKey(keyboardEvent, key);
  }
}
