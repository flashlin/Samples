import { describe } from "vitest"
import { required } from '../validator'

describe('required', () => {
  it.each([
    [undefined],
    [null],
    [''],
    [' '],
  ])('should return false for %s', ( value: any ) => {
    expect(required(value)).toBeFalsy()
  })

  it.each(
    [
      ['a'],
      ['ab'],
    ])('should return true for %s', () => {
    expect(required('a')).toBeTruthy()
  })
})