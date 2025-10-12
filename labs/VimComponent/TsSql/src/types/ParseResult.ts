import { ParseError } from './ParseError';
import { Expression } from './BaseExpression';

// Parse result containing partial expression and errors
export interface ParseResult<T extends Expression = Expression> {
  result: T;
  errors: ParseError[];
}

// Helper to check if parse was successful
export function isSuccessful(result: ParseResult): boolean {
  return result.errors.length === 0;
}

// Helper to create successful result
export function success<T extends Expression>(expr: T): ParseResult<T> {
  return {
    result: expr,
    errors: []
  };
}

// Helper to create error result
export function failure<T extends Expression>(expr: T, ...errors: ParseError[]): ParseResult<T> {
  return {
    result: expr,
    errors
  };
}

