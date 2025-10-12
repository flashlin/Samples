import { BaseExpression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// Literal value expression
export class LiteralExpression extends BaseExpression {
  constructor(
    public readonly value: string | number | boolean | null,
    public readonly literalType: 'string' | 'number' | 'boolean' | 'null' = 'string'
  ) {
    super(ExpressionType.Literal);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitLiteral(this);
  }
}

