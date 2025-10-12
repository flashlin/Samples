import { ExpressionVisitor } from '../types/ExpressionVisitor';
import { Expression, BaseExpression } from '../types/BaseExpression';
import { BinaryOperator } from '../types/ExpressionType';

// LINQ expressions
import { LinqQueryExpression } from '../linqExpressions/LinqQueryExpression';
import { LinqFromExpression } from '../linqExpressions/LinqFromExpression';
import { LinqJoinExpression } from '../linqExpressions/LinqJoinExpression';
import { LinqWhereExpression } from '../linqExpressions/LinqWhereExpression';
import { LinqGroupByExpression } from '../linqExpressions/LinqGroupByExpression';
import { LinqHavingExpression } from '../linqExpressions/LinqHavingExpression';
import { LinqOrderByExpression } from '../linqExpressions/LinqOrderByExpression';
import { LinqSelectExpression } from '../linqExpressions/LinqSelectExpression';

// T-SQL expressions
import { QueryExpression } from '../expressions/QueryExpression';
import { FromExpression } from '../expressions/FromExpression';
import { JoinExpression } from '../expressions/JoinExpression';
import { WhereExpression } from '../expressions/WhereExpression';
import { GroupByExpression } from '../expressions/GroupByExpression';
import { HavingExpression } from '../expressions/HavingExpression';
import { OrderByExpression, OrderByItem } from '../expressions/OrderByExpression';
import { SelectExpression, SelectItem } from '../expressions/SelectExpression';
import { ColumnExpression } from '../expressions/ColumnExpression';
import { LiteralExpression } from '../expressions/LiteralExpression';
import { BinaryExpression } from '../expressions/BinaryExpression';
import { UnaryExpression } from '../expressions/UnaryExpression';
import { FunctionExpression } from '../expressions/FunctionExpression';

// Converter from LINQ expressions to T-SQL expressions
export class LinqToTSqlConverter {
  
  // Convert LINQ query to T-SQL query
  convert(linqQuery: LinqQueryExpression): QueryExpression {
    // Convert FROM
    const from = linqQuery.from 
      ? new FromExpression(linqQuery.from.tableName, linqQuery.from.alias)
      : undefined;
    
    // Convert JOINs
    const joins = linqQuery.joins.map(j => 
      new JoinExpression(j.joinType, j.tableName, this.convertExpression(j.condition), j.alias)
    );
    
    // Convert WHEREs - combine multiple WHERE clauses with AND
    let where: WhereExpression | undefined;
    if (linqQuery.wheres.length > 0) {
      let combinedCondition: Expression = this.convertExpression(linqQuery.wheres[0].condition);
      
      for (let i = 1; i < linqQuery.wheres.length; i++) {
        const nextCondition = this.convertExpression(linqQuery.wheres[i].condition);
        combinedCondition = new BinaryExpression(combinedCondition, BinaryOperator.And, nextCondition);
      }
      
      where = new WhereExpression(combinedCondition);
    }
    
    // Convert GROUP BYs - combine multiple GROUP BY clauses
    let groupBy: GroupByExpression | undefined;
    if (linqQuery.groupBys.length > 0) {
      const allColumns: Expression[] = [];
      for (const gb of linqQuery.groupBys) {
        allColumns.push(...gb.columns.map(c => this.convertExpression(c)));
      }
      groupBy = new GroupByExpression(allColumns);
    }
    
    // Convert HAVING
    const having = linqQuery.having 
      ? new HavingExpression(this.convertExpression(linqQuery.having.condition))
      : undefined;
    
    // Convert ORDER BYs - combine multiple ORDER BY clauses
    let orderBy: OrderByExpression | undefined;
    if (linqQuery.orderBys.length > 0) {
      const allItems: OrderByItem[] = [];
      for (const ob of linqQuery.orderBys) {
        allItems.push(...ob.items.map(item => ({
          expression: this.convertExpression(item.expression),
          direction: item.direction
        })));
      }
      orderBy = new OrderByExpression(allItems);
    }
    
    // Convert SELECT
    const select = linqQuery.select 
      ? new SelectExpression(
          linqQuery.select.items.map(item => ({
            expression: this.convertExpression(item.expression),
            alias: item.alias
          })),
          linqQuery.select.isDistinct
        )
      : undefined;
    
    return new QueryExpression(select, from, joins, where, groupBy, having, orderBy);
  }
  
  // Convert individual expression (recursive)
  private convertExpression(expr: Expression): Expression {
    // Basic expressions don't need conversion, they're shared
    if (expr instanceof ColumnExpression ||
        expr instanceof LiteralExpression ||
        expr instanceof FunctionExpression) {
      return expr;
    }
    
    // Binary expressions - recursively convert operands
    if (expr instanceof BinaryExpression) {
      return new BinaryExpression(
        this.convertExpression(expr.left),
        expr.operator,
        this.convertExpression(expr.right)
      );
    }
    
    // Unary expressions - recursively convert operand
    if (expr instanceof UnaryExpression) {
      return new UnaryExpression(
        expr.operator,
        this.convertExpression(expr.operand)
      );
    }
    
    // Default: return as-is
    return expr;
  }
}

