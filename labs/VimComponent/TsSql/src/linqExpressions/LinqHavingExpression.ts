import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// LINQ HAVING expression
export class LinqHavingExpression extends BaseExpression {
  constructor(
    public readonly condition: Expression
  ) {
    super(ExpressionType.LinqHaving);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitLinqHaving(this);
  }
}

