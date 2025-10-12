// Parse error information
export class ParseError {
  constructor(
    public readonly message: string,
    public readonly position: number,
    public readonly line: number,
    public readonly column: number,
    public readonly errorCode?: string
  ) {}
  
  toString(): string {
    return `Parse error at line ${this.line}, column ${this.column} (position ${this.position}): ${this.message}${this.errorCode ? ` [${this.errorCode}]` : ''}`;
  }
}

