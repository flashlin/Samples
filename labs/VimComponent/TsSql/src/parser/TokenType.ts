// Token types for SQL/LINQ parsing
export enum TokenType {
  // Keywords
  FROM = 'FROM',
  JOIN = 'JOIN',
  INNER = 'INNER',
  LEFT = 'LEFT',
  RIGHT = 'RIGHT',
  FULL = 'FULL',
  CROSS = 'CROSS',
  ON = 'ON',
  WHERE = 'WHERE',
  GROUP = 'GROUP',
  BY = 'BY',
  HAVING = 'HAVING',
  ORDER = 'ORDER',
  SELECT = 'SELECT',
  DISTINCT = 'DISTINCT',
  TOP = 'TOP',
  AS = 'AS',
  WITH = 'WITH',
  DROP = 'DROP',
  TABLE = 'TABLE',
  
  // Logical operators
  AND = 'AND',
  OR = 'OR',
  NOT = 'NOT',
  
  // Comparison operators
  EQUAL = '=',
  NOT_EQUAL = '<>',
  GREATER_THAN = '>',
  LESS_THAN = '<',
  GREATER_EQUAL = '>=',
  LESS_EQUAL = '<=',
  LIKE = 'LIKE',
  IN = 'IN',
  IS = 'IS',
  NULL = 'NULL',
  EXISTS = 'EXISTS',
  
  // Arithmetic operators
  PLUS = '+',
  MINUS = '-',
  MULTIPLY = '*',
  DIVIDE = '/',
  MODULO = '%',
  
  // Delimiters
  LEFT_PAREN = '(',
  RIGHT_PAREN = ')',
  COMMA = ',',
  DOT = '.',
  
  // Literals
  IDENTIFIER = 'IDENTIFIER',
  NUMBER = 'NUMBER',
  STRING = 'STRING',
  
  // Special
  ASC = 'ASC',
  DESC = 'DESC',
  
  // End of input
  EOF = 'EOF',
}

// Token class
export class Token {
  constructor(
    public readonly type: TokenType,
    public readonly value: string,
    public readonly position: number,
    public readonly line: number,
    public readonly column: number
  ) {}
  
  toString(): string {
    return `Token(${this.type}, "${this.value}", line: ${this.line}, col: ${this.column})`;
  }
}

