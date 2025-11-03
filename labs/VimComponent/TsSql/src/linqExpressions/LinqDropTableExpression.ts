import { BaseExpression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

export class LinqDropTableExpression extends BaseExpression {
  constructor(
    public readonly tableName: string,
    public readonly databaseName?: string,
    public readonly schemaName?: string
  ) {
    super(ExpressionType.LinqDropTable);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitLinqDropTable(this);
  }
}

