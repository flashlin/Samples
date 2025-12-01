import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

export class ArrayExpression extends BaseExpression {
  constructor(
    public readonly elements: Expression[]
  ) {
    super(ExpressionType.Array);
  }

  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitArray(this);
  }
}
