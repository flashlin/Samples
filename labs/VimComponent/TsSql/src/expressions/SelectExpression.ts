import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// Select item
export interface SelectItem {
  expression: Expression;
  alias?: string;
}

// SELECT clause expression
export class SelectExpression extends BaseExpression {
  constructor(
    public readonly items: SelectItem[],
    public readonly isDistinct: boolean = false,
    public readonly topCount?: number | Expression
  ) {
    super(ExpressionType.Select);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitSelect(this);
  }
  
  // Helper to check if selecting all columns
  get isSelectAll(): boolean {
    return this.items.length === 1 && 
           this.items[0].expression.type === ExpressionType.Column &&
           (this.items[0].expression as any).columnName === '*';
  }
}

