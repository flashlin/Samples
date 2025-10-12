import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType, OrderDirection } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// Single order by item
export interface LinqOrderByItem {
  expression: Expression;
  direction: OrderDirection;
}

// LINQ ORDER BY expression
export class LinqOrderByExpression extends BaseExpression {
  constructor(
    public readonly items: LinqOrderByItem[]
  ) {
    super(ExpressionType.LinqOrderBy);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitLinqOrderBy(this);
  }
}

