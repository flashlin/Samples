import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType, UnaryOperator } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// Unary operation expression
export class UnaryExpression extends BaseExpression {
  constructor(
    public readonly operator: UnaryOperator,
    public readonly operand: Expression
  ) {
    super(ExpressionType.Unary);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitUnary(this);
  }
}

