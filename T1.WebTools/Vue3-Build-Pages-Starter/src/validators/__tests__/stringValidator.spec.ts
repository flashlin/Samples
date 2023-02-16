import { describe, it } from 'vitest'
import { useStringValidator } from '../stringValidator'

describe('StringLengthValidator tests', () => {
  it.each([
    ['test'],
    ['tes'],
    ['12'],
    ['null'],
  ])('given max length 4 when input is %s should is valid', ( str ) => {
    expect(useStringValidator().length().max(4).isValid(str)).toBeTruthy()
  })
  it.each([
    [' test'],
    ['test_'],
    ['test_1'],
    ['undefined'],
  ])('given max length 4 when input is %s should is invalid', ( str ) => {
    expect(useStringValidator().length().max(4).isValid(str)).toBeFalsy()
  })

  it.each([
    ['abc'],
    ['abcd'],
    ['null'],
    ['undefined'],
  ])('given min length 3 when input is %s should is valid', ( str ) => {
    expect(useStringValidator().length().min(3).isValid(str)).toBeTruthy()
  })

  it.each([
    ['ab'],
    ['a'],
    [''],
  ])('given min length 3 when input is %s should is invalid', ( str ) => {
    expect(useStringValidator().length().min(3).isValid(str)).toBeFalsy()
  })

  it.each([
    ['abc'],
    ['abcd'],
    ['abcde'],
    [' ab'],
    ['ab '],
    ['abcd '],
    [' abcd'],
  ])('given min length 3 and max length 5 when input is %s should is valid', ( str ) => {
    expect(useStringValidator().length().min(3).max(5).isValid(str)).toBeTruthy()
  })

  it.each([
    ['ab'],
    ['a'],
    [' a'],
    ['abcdef'],
    ['abcdefg'],
    [' abcdef'],
  ])('given min length 3 and max length 5 when input is %s should is invalid', ( str ) => {
    expect(useStringValidator().length().min(3).max(5).isValid(str)).toBeFalsy()
  })
  it.each([
    [3],
    [4],
    [5],
  ])('given min length 3 and max length 5 when input is %i space should is valid', ( spaceNumber ) => {
    expect(useStringValidator().length().min(3).max(5).isValid(' '.repeat(spaceNumber))).toBeTruthy()
  })

  it.each([
    [2],
    [1],
    [6],
    [7],
  ])('given min length 3 and max length 5 when input is %i space should is invalid', ( spaceNumber ) => {
    expect(useStringValidator().length().min(3).max(5).isValid(' '.repeat(spaceNumber))).toBeFalsy()
  })

  it('given a number should be invalid', () => {
    expect(useStringValidator().length().isValid(1)).toBeFalsy()
    expect(useStringValidator().length().isValid(12)).toBeFalsy()
  })
})

describe('StringContainWhiteSpaceValidator tests', () => {
  it.each([
    [''],
    ['t'],
    ['t_'],
    ['test'],
    ['@test'],
  ])('should be valid', ( str ) => {
    expect(useStringValidator().notContainWhiteSpace().isValid(str)).toBeTruthy()
  })

  it.each([
    [' '],
    [' test'],
    ['test '],
    [' test '],
  ])('should be invalid', ( str ) => {
    expect(useStringValidator().notContainWhiteSpace().isValid(str)).toBeFalsy()
  })
})

describe('StringIsNotEmptyOrWhiteSpaceValidator tests', () => {
  it.each([
    ['a'],
    ['ab'],
    ['@ab'],
    ['_ab'],
  ])('should be valid', ( str ) => {
    expect(useStringValidator().notEmptyOrWhiteSpace().isValid(str)).toBeTruthy()
  })

  it.each([
    [''],
    [' '],
    ['test '],
    [' test'],
    [' test '],
  ])('should be invalid', ( str ) => {
    expect(useStringValidator().notEmptyOrWhiteSpace().isValid(str)).toBeFalsy()
  })
})

describe('StringNotContainNumberValidator tests', () => {
  it.each([
    ['a'],
    ['abc'],
    ['a_c'],
    ['a@c'],
    [''],
  ])('should be valid', ( str ) => {
    expect(useStringValidator().notContainNumber().isValid(str)).toBeTruthy()
  })

  it.each([
    ['1a'],
    ['a1'],
    ['a1a'],
    ['1a1'],
  ])('should be invalid', ( str ) => {
    expect(useStringValidator().notContainNumber().isValid(str)).toBeFalsy()
  })
})
