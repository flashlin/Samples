import { describe, it } from 'vitest'
import { useDateValidator } from 'src/validators/dateValidator'

describe('dateValidator tests', () => {
  it.each([
    ['1'],
    ['31'],
  ])('%s should be valid day', ( day: unknown ) => {
    expect(useDateValidator().day().isValid(day)).toBeTruthy()
  })
  it.each([
    ['0'],
    ['32'],
  ])('%s should be invalid day', ( day: unknown ) => {
    expect(useDateValidator().day().isValid(day)).toBeFalsy()
  })
  it.each([
    ['1'],
    ['12'],
  ])('%s should be valid month', ( day: unknown ) => {
    expect(useDateValidator().month().isValid(day)).toBeTruthy()
  })
  it.each([
    ['0'],
    ['13'],
  ])('%s should be invalid month', ( day: unknown ) => {
    expect(useDateValidator().month().isValid(day)).toBeFalsy()
  })
})