import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// Function call expression
export class FunctionExpression extends BaseExpression {
  constructor(
    public readonly functionName: string,
    public readonly args: Expression[],
    public readonly alias?: string
  ) {
    super(ExpressionType.Function);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitFunction(this);
  }
}

