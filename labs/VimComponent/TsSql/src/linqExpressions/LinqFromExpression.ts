import { BaseExpression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// LINQ FROM expression (at the beginning)
export class LinqFromExpression extends BaseExpression {
  constructor(
    public readonly tableName: string,
    public readonly alias?: string,
    public readonly databaseName?: string
  ) {
    super(ExpressionType.LinqFrom);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitLinqFrom(this);
  }
}

