import { ExpressionType } from './ExpressionType';
import { ExpressionVisitor } from './ExpressionVisitor';

// Base interface for all expressions
export interface Expression {
  readonly type: ExpressionType;
  accept<T>(visitor: ExpressionVisitor<T>): T;
}

// Base abstract class for all expressions
export abstract class BaseExpression implements Expression {
  constructor(public readonly type: ExpressionType) {}
  
  abstract accept<T>(visitor: ExpressionVisitor<T>): T;
}

