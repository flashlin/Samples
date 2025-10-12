import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType, BinaryOperator } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// Binary operation expression
export class BinaryExpression extends BaseExpression {
  constructor(
    public readonly left: Expression,
    public readonly operator: BinaryOperator,
    public readonly right: Expression
  ) {
    super(ExpressionType.Binary);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitBinary(this);
  }
}

