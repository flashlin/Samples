import { describe, it } from 'vitest'
import { useEmailValidator } from '../emailValidator'

describe('email validator tests', () => {
  it.each([
    ['someone@example.com'],
    ['someone@example.co'],
    ['someone123@example.co.uk'],
    ['Pelé@example.com'],
    ['very.common@example.com'],
    ['other.email-with-dash@example.com'],
    ['disposable.style.email.with+symbol@example.com'],
    ['disposable.style.email.with+symbol@example.com'],
  ])('%s should be valid', ( email ) => {
    expect(useEmailValidator().isValidFormat().isValid(email)).toBeTruthy()
  })

  it.each([
    [''],
    [null],
    [undefined],
    [[]],
    ['undefined'],
    ['null'],
    ['@example.com'],
    ['@example'],
    ['someone@example.c'],
  ])('%s should be invalid', (email) => {
    expect(useEmailValidator().isValidFormat().isValid(email)).toBeFalsy()
  })
})
