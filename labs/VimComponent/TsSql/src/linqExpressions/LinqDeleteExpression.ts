import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

export class LinqDeleteExpression extends BaseExpression {
  constructor(
    public readonly tableName: string,
    public readonly whereCondition?: Expression,
    public readonly databaseName?: string,
    public readonly topCount?: number,
    public readonly isPercent?: boolean,
    public readonly schemaName?: string
  ) {
    super(ExpressionType.LinqDelete);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitLinqDelete(this);
  }
}

