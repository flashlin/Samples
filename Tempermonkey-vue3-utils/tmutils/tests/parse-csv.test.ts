import { describe, expect, it } from 'vitest';
import "../src/models/csv-parser";

describe('parse csv test', () => {
  it('header2 contain ,', () => {
    let actual = `id,"name,birth"`.csvSplit();
    expect(actual).toStrictEqual([
      'id',
      'name,birth'
    ]);
  });


  it('header1 contain ,', () => {
    let actual = `"id, 1",name`.csvSplit();
    expect(actual).toStrictEqual([
      'id, 1',
      'name'
    ]);
  });
});