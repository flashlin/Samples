import { textTokenize } from '../textTokenizer'

describe('textTokenize', () => {
  // 測試數字
  it('should tokenize numbers', () => {
    const tokens = Array.from(textTokenize('123 456'))
    expect(tokens).toEqual(['123', ' ', '456'])
  })

  // 測試小數點
  it('should tokenize decimal numbers as single tokens', () => {
    const tokens = Array.from(textTokenize('3.14 0.99'))
    // 預期: 3.14,  , 0.99
    expect(tokens).toEqual(['3.14', ' ', '0.99'])
  })

  // 測試特殊符號
  it('should tokenize special symbols', () => {
    const tokens = Array.from(textTokenize('a+b-c!=d'))
    expect(tokens).toEqual(['a', '+', 'b', '-', 'c', '!=', 'd'])
  })

  // 測試字串
  it('should tokenize quoted strings', () => {
    const tokens = Array.from(textTokenize('say "hello world"'))
    expect(tokens).toEqual(['say', ' ', '"hello world"'])
  })

  // 測試混合案例
  it('should tokenize mixed input', () => {
    const tokens = Array.from(textTokenize('x = 42 + "foo"'))
    expect(tokens).toEqual(['x', ' ', '=', ' ', '42', ' ', '+', ' ', '"foo"'])
  })

  // 測試單引號字串
  it('should tokenize single quoted strings', () => {
    const tokens = Array.from(textTokenize("say 'hello world'"))
    expect(tokens).toEqual(['say', ' ', "'hello world'"])
  })
}) 