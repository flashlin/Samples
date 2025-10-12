import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// WHERE clause expression
export class WhereExpression extends BaseExpression {
  constructor(
    public readonly condition: Expression
  ) {
    super(ExpressionType.Where);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitWhere(this);
  }
}

