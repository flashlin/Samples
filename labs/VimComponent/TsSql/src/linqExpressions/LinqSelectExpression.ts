import { BaseExpression, Expression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';

// Select item
export interface LinqSelectItem {
  expression: Expression;
  alias?: string;
}

// LINQ SELECT expression (at the end)
export class LinqSelectExpression extends BaseExpression {
  constructor(
    public readonly items: LinqSelectItem[],
    public readonly isDistinct: boolean = false
  ) {
    super(ExpressionType.LinqSelect);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitLinqSelect(this);
  }
  
  // Helper to check if selecting all columns
  get isSelectAll(): boolean {
    return this.items.length === 1 && 
           this.items[0].expression.type === ExpressionType.Column &&
           (this.items[0].expression as any).columnName === '*';
  }
}

