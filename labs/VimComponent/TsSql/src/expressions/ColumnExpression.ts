import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// Column reference expression
export class ColumnExpression extends BaseExpression {
  constructor(
    public readonly columnName: string,
    public readonly tableName?: string,
    public readonly alias?: string
  ) {
    super(ExpressionType.Column);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitColumn(this);
  }
  
  get fullName(): string {
    return this.tableName ? `${this.tableName}.${this.columnName}` : this.columnName;
  }
}

