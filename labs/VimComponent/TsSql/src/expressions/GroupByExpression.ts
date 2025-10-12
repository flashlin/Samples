import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// GROUP BY clause expression
export class GroupByExpression extends BaseExpression {
  constructor(
    public readonly columns: Expression[]
  ) {
    super(ExpressionType.GroupBy);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitGroupBy(this);
  }
}

