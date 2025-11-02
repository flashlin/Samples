import { ParseError } from './ParseError';
import { LinqStatement } from './StatementTypes';

// Parse result containing partial expression and errors
export interface ParseResult {
  result: LinqStatement;
  errors: ParseError[];
}

// Helper to check if parse was successful
export function isSuccessful(result: ParseResult): boolean {
  return result.errors.length === 0;
}

// Helper to create successful result
export function success(expr: LinqStatement): ParseResult {
  return {
    result: expr,
    errors: []
  };
}

// Helper to create error result
export function failure(expr: LinqStatement, ...errors: ParseError[]): ParseResult {
  return {
    result: expr,
    errors
  };
}

