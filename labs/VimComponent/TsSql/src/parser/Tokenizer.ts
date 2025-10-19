import { Token, TokenType } from './TokenType';
import { ParseError } from '../types/ParseError';

// Keyword mapping
const KEYWORDS: Record<string, TokenType> = {
  'FROM': TokenType.FROM,
  'JOIN': TokenType.JOIN,
  'INNER': TokenType.INNER,
  'LEFT': TokenType.LEFT,
  'RIGHT': TokenType.RIGHT,
  'FULL': TokenType.FULL,
  'CROSS': TokenType.CROSS,
  'ON': TokenType.ON,
  'WHERE': TokenType.WHERE,
  'GROUP': TokenType.GROUP,
  'BY': TokenType.BY,
  'HAVING': TokenType.HAVING,
  'ORDER': TokenType.ORDER,
  'SELECT': TokenType.SELECT,
  'DISTINCT': TokenType.DISTINCT,
  'AS': TokenType.AS,
  'WITH': TokenType.WITH,
  'AND': TokenType.AND,
  'OR': TokenType.OR,
  'NOT': TokenType.NOT,
  'LIKE': TokenType.LIKE,
  'IN': TokenType.IN,
  'IS': TokenType.IS,
  'NULL': TokenType.NULL,
  'EXISTS': TokenType.EXISTS,
  'ASC': TokenType.ASC,
  'DESC': TokenType.DESC,
};

// Tokenizer class
export class Tokenizer {
  private input: string;
  private position: number = 0;
  private line: number = 1;
  private column: number = 1;
  private tokens: Token[] = [];
  private errors: ParseError[] = [];
  
  constructor(input: string) {
    this.input = input;
  }
  
  // Tokenize the input
  tokenize(): { tokens: Token[]; errors: ParseError[] } {
    while (this.position < this.input.length) {
      this.skipWhitespace();
      
      if (this.position >= this.input.length) {
        break;
      }
      
      const char = this.current();
      
      // Skip comments
      if (char === '-' && this.peek() === '-') {
        this.skipLineComment();
        continue;
      }
      
      if (char === '/' && this.peek() === '*') {
        this.skipBlockComment();
        continue;
      }
      
      // String literals
      if (char === "'" || char === '"') {
        this.scanString(char);
        continue;
      }
      
      // Numbers
      if (this.isDigit(char)) {
        this.scanNumber();
        continue;
      }
      
      // Identifiers and keywords
      if (this.isAlpha(char)) {
        this.scanIdentifier();
        continue;
      }
      
      // Operators and delimiters
      if (this.scanOperator()) {
        continue;
      }
      
      // Unknown character
      this.errors.push(new ParseError(
        `Unexpected character: '${char}'`,
        this.position,
        this.line,
        this.column,
        'UNEXPECTED_CHAR'
      ));
      this.advance();
    }
    
    // Add EOF token
    this.tokens.push(new Token(
      TokenType.EOF,
      '',
      this.position,
      this.line,
      this.column
    ));
    
    return { tokens: this.tokens, errors: this.errors };
  }
  
  private current(): string {
    return this.input[this.position];
  }
  
  private peek(offset: number = 1): string {
    const pos = this.position + offset;
    return pos < this.input.length ? this.input[pos] : '';
  }
  
  private advance(): string {
    const char = this.current();
    this.position++;
    if (char === '\n') {
      this.line++;
      this.column = 1;
    } else {
      this.column++;
    }
    return char;
  }
  
  private skipWhitespace(): void {
    while (this.position < this.input.length && this.isWhitespace(this.current())) {
      this.advance();
    }
  }
  
  private skipLineComment(): void {
    // Skip '--'
    this.advance();
    this.advance();
    
    // Skip until end of line
    while (this.position < this.input.length && this.current() !== '\n') {
      this.advance();
    }
  }
  
  private skipBlockComment(): void {
    // Skip '/*'
    this.advance();
    this.advance();
    
    // Skip until '*/'
    while (this.position < this.input.length - 1) {
      if (this.current() === '*' && this.peek() === '/') {
        this.advance(); // Skip '*'
        this.advance(); // Skip '/'
        break;
      }
      this.advance();
    }
  }
  
  private scanString(quote: string): void {
    const startPos = this.position;
    const startLine = this.line;
    const startCol = this.column;
    
    this.advance(); // Skip opening quote
    
    let value = '';
    while (this.position < this.input.length && this.current() !== quote) {
      if (this.current() === '\\') {
        this.advance();
        if (this.position < this.input.length) {
          value += this.advance();
        }
      } else {
        value += this.advance();
      }
    }
    
    if (this.position >= this.input.length) {
      this.errors.push(new ParseError(
        'Unterminated string literal',
        startPos,
        startLine,
        startCol,
        'UNTERMINATED_STRING'
      ));
    } else {
      this.advance(); // Skip closing quote
    }
    
    this.tokens.push(new Token(
      TokenType.STRING,
      value,
      startPos,
      startLine,
      startCol
    ));
  }
  
  private scanNumber(): void {
    const startPos = this.position;
    const startLine = this.line;
    const startCol = this.column;
    
    let value = '';
    while (this.position < this.input.length && (this.isDigit(this.current()) || this.current() === '.')) {
      value += this.advance();
    }
    
    this.tokens.push(new Token(
      TokenType.NUMBER,
      value,
      startPos,
      startLine,
      startCol
    ));
  }
  
  private scanIdentifier(): void {
    const startPos = this.position;
    const startLine = this.line;
    const startCol = this.column;
    
    let value = '';
    while (this.position < this.input.length && (this.isAlphaNumeric(this.current()) || this.current() === '_')) {
      value += this.advance();
    }
    
    const upperValue = value.toUpperCase();
    const tokenType = KEYWORDS[upperValue] || TokenType.IDENTIFIER;
    
    this.tokens.push(new Token(
      tokenType,
      value,
      startPos,
      startLine,
      startCol
    ));
  }
  
  private scanOperator(): boolean {
    const startPos = this.position;
    const startLine = this.line;
    const startCol = this.column;
    const char = this.current();
    
    // Two-character operators
    if (char === '<' && this.peek() === '>') {
      this.advance();
      this.advance();
      this.tokens.push(new Token(TokenType.NOT_EQUAL, '<>', startPos, startLine, startCol));
      return true;
    }
    
    if (char === '<' && this.peek() === '=') {
      this.advance();
      this.advance();
      this.tokens.push(new Token(TokenType.LESS_EQUAL, '<=', startPos, startLine, startCol));
      return true;
    }
    
    if (char === '>' && this.peek() === '=') {
      this.advance();
      this.advance();
      this.tokens.push(new Token(TokenType.GREATER_EQUAL, '>=', startPos, startLine, startCol));
      return true;
    }
    
    // Single-character operators
    const singleCharTokens: Record<string, TokenType> = {
      '=': TokenType.EQUAL,
      '<': TokenType.LESS_THAN,
      '>': TokenType.GREATER_THAN,
      '+': TokenType.PLUS,
      '-': TokenType.MINUS,
      '*': TokenType.MULTIPLY,
      '/': TokenType.DIVIDE,
      '%': TokenType.MODULO,
      '(': TokenType.LEFT_PAREN,
      ')': TokenType.RIGHT_PAREN,
      ',': TokenType.COMMA,
      '.': TokenType.DOT,
    };
    
    if (char in singleCharTokens) {
      this.advance();
      this.tokens.push(new Token(singleCharTokens[char], char, startPos, startLine, startCol));
      return true;
    }
    
    return false;
  }
  
  private isWhitespace(char: string): boolean {
    return /\s/.test(char);
  }
  
  private isDigit(char: string): boolean {
    return /[0-9]/.test(char);
  }
  
  private isAlpha(char: string): boolean {
    return /[a-zA-Z]/.test(char);
  }
  
  private isAlphaNumeric(char: string): boolean {
    return /[a-zA-Z0-9]/.test(char);
  }
}

