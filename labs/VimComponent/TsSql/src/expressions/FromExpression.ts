import { BaseExpression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// FROM clause expression
export class FromExpression extends BaseExpression {
  constructor(
    public readonly tableName: string,
    public readonly alias?: string,
    public readonly hints?: string[]
  ) {
    super(ExpressionType.From);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitFrom(this);
  }
}

