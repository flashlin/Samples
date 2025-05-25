// Tokenize text by rules: numbers, quoted strings, special symbols, spaces, newlines
export function* textTokenize(text: string): IterableIterator<string> {
  let i = 0
  const len = text.length
  while (i < len) {
    const c = text[i]
    // 0. Identifier: starts with _ or letter, then _/letter/number
    if (c === '_' || /[a-zA-Z]/.test(c)) {
      let start = i
      i++
      while (i < len && (text[i] === '_' || /[a-zA-Z0-9]/.test(text[i]))) i++
      yield text.slice(start, i)
      continue
    }
    // 1. Numbers (float or integer)
    if (/[0-9]/.test(c)) {
      let start = i
      while (i < len && /[0-9]/.test(text[i])) i++
      // check for float
      if (i < len && text[i] === '.' && i + 1 < len && /[0-9]/.test(text[i + 1])) {
        i++ // skip '.'
        while (i < len && /[0-9]/.test(text[i])) i++
      }
      yield text.slice(start, i)
      continue
    }
    // 2. Quoted string (double quote)
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
    // 2. Quoted string (single quote)
    if (c === "'") {
      i++
      let str = "'"
      while (i < len) {
        if (text[i] === '\\' && text[i + 1] === "'") {
          str += "\\'"
          i += 2
        } else if (text[i] === "'") {
          str += "'"
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