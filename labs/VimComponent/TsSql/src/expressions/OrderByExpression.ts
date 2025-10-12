import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType, OrderDirection } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// Single order by item
export interface OrderByItem {
  expression: Expression;
  direction: OrderDirection;
}

// ORDER BY clause expression
export class OrderByExpression extends BaseExpression {
  constructor(
    public readonly items: OrderByItem[]
  ) {
    super(ExpressionType.OrderBy);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitOrderBy(this);
  }
}

