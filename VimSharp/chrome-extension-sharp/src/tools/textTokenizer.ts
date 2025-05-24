// Tokenize text by rules: numbers, quoted strings, special symbols, spaces, newlines
export function* textTokenize(text: string): IterableIterator<string> {
  let i = 0
  const len = text.length
  while (i < len) {
    const c = text[i]
    // 1. Numbers
    if (/[0-9]/.test(c)) {
      let start = i
      while (i < len && /[0-9]/.test(text[i])) i++
      yield text.slice(start, i)
      continue
    }
    // 2. Quoted string
    if (c === '"') {
      i++
      let str = '"'
      while (i < len) {
        if (text[i] === '\\' && text[i + 1] === '"') {
          str += '\\"'
          i += 2
        } else if (text[i] === '"') {
          str += '"'
          i++
          break
        } else {
          str += text[i]
          i++
        }
      }
      yield str
      continue
    }
    // 3. Special symbols
    if (/[><=~!\+\-]/.test(c)) {
      let start = i
      while (i < len && /[><=~!\+\-]/.test(text[i])) i++
      yield text.slice(start, i)
      continue
    }
    // 4. Spaces (including tab)
    if (c === ' ' || c === '\t') {
      let start = i
      while (i < len && (text[i] === ' ' || text[i] === '\t')) i++
      yield text.slice(start, i)
      continue
    }
    // 5. Newline (\r\n)
    if (c === '\r' && text[i + 1] === '\n') {
      yield '\r\n'
      i += 2
      continue
    }
    // Single char fallback
    yield c
    i++
  }
} 