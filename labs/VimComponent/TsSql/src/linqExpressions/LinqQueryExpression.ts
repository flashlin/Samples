import { BaseExpression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';
import { LinqFromExpression } from './LinqFromExpression';
import { LinqJoinExpression } from './LinqJoinExpression';
import { LinqWhereExpression } from './LinqWhereExpression';
import { LinqGroupByExpression } from './LinqGroupByExpression';
import { LinqHavingExpression } from './LinqHavingExpression';
import { LinqOrderByExpression } from './LinqOrderByExpression';
import { LinqSelectExpression } from './LinqSelectExpression';

// Complete LINQ query expression (FROM-first syntax)
// Order: FROM → JOIN → WHERE → GROUP BY → HAVING → ORDER BY → SELECT
export class LinqQueryExpression extends BaseExpression {
  constructor(
    public readonly from?: LinqFromExpression,
    public readonly joins: LinqJoinExpression[] = [],
    public readonly wheres: LinqWhereExpression[] = [],
    public readonly groupBys: LinqGroupByExpression[] = [],
    public readonly having?: LinqHavingExpression,
    public readonly orderBys: LinqOrderByExpression[] = [],
    public readonly select?: LinqSelectExpression
  ) {
    super(ExpressionType.LinqQuery);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitLinqQuery(this);
  }
  
  // Check if query is complete
  get isComplete(): boolean {
    return this.from !== undefined && this.select !== undefined;
  }
  
  // Check if query is empty
  get isEmpty(): boolean {
    return !this.from && this.joins.length === 0 && this.wheres.length === 0 &&
           this.groupBys.length === 0 && !this.having && this.orderBys.length === 0 &&
           !this.select;
  }
}

