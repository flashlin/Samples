import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType, JoinType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// JOIN clause expression
export class JoinExpression extends BaseExpression {
  constructor(
    public readonly joinType: JoinType,
    public readonly tableName: string,
    public readonly condition: Expression,
    public readonly alias?: string
  ) {
    super(ExpressionType.Join);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitJoin(this);
  }
}

