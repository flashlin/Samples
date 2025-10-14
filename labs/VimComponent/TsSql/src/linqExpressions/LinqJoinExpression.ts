import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType, JoinType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// LINQ JOIN expression
export class LinqJoinExpression extends BaseExpression {
  constructor(
    public readonly joinType: JoinType,
    public readonly tableName: string,
    public readonly condition: Expression,
    public readonly alias?: string,
    public readonly databaseName?: string
  ) {
    super(ExpressionType.LinqJoin);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitLinqJoin(this);
  }
}

