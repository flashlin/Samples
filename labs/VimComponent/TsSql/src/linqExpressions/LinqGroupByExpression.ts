import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// LINQ GROUP BY expression
export class LinqGroupByExpression extends BaseExpression {
  constructor(
    public readonly columns: Expression[]
  ) {
    super(ExpressionType.LinqGroupBy);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitLinqGroupBy(this);
  }
}

