import { BaseExpression } from '../types/BaseExpression';
import { ExpressionType } from '../types/ExpressionType';
import { ExpressionVisitor } from '../types/ExpressionVisitor';
import { SelectExpression } from './SelectExpression';
import { FromExpression } from './FromExpression';
import { JoinExpression } from './JoinExpression';
import { WhereExpression } from './WhereExpression';
import { GroupByExpression } from './GroupByExpression';
import { HavingExpression } from './HavingExpression';
import { OrderByExpression } from './OrderByExpression';

// Complete T-SQL query expression
export class QueryExpression extends BaseExpression {
  constructor(
    public readonly select?: SelectExpression,
    public readonly from?: FromExpression,
    public readonly joins: JoinExpression[] = [],
    public readonly where?: WhereExpression,
    public readonly groupBy?: GroupByExpression,
    public readonly having?: HavingExpression,
    public readonly orderBy?: OrderByExpression
  ) {
    super(ExpressionType.Query);
  }
  
  accept<T>(visitor: ExpressionVisitor<T>): T {
    return visitor.visitQuery(this);
  }
  
  // Check if query is complete
  get isComplete(): boolean {
    return this.select !== undefined && this.from !== undefined;
  }
  
  // Check if query is empty
  get isEmpty(): boolean {
    return !this.select && !this.from && this.joins.length === 0 && 
           !this.where && !this.groupBy && !this.having && !this.orderBy;
  }
}

