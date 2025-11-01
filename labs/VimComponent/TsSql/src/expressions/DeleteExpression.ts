import { BaseExpression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';
import { WhereExpression } from './WhereExpression';

export class DeleteExpression extends BaseExpression {
  constructor(
    public readonly tableName: string,
    public readonly where?: WhereExpression,
    public readonly topCount?: number,
    public readonly isPercent?: boolean
  ) {
    super(ExpressionType.Delete);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitDelete(this);
  }
}

