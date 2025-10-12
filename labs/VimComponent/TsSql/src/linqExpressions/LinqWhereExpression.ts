import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// LINQ WHERE expression
export class LinqWhereExpression extends BaseExpression {
  constructor(
    public readonly condition: Expression
  ) {
    super(ExpressionType.LinqWhere);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitLinqWhere(this);
  }
}

