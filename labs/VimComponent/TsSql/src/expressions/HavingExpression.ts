import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// HAVING clause expression
export class HavingExpression extends BaseExpression {
  constructor(
    public readonly condition: Expression
  ) {
    super(ExpressionType.Having);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitHaving(this);
  }
}

