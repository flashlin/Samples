import { BaseExpression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

export class DropTableExpression extends BaseExpression {
  constructor(
    public readonly tableName: string
  ) {
    super(ExpressionType.DropTable);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitDropTable(this);
  }
}

