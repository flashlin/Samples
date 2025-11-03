import { ExpressionVisitor } from '../types/ExpressionVisitor';
import { Expression, BaseExpression } from '../types/BaseExpression';
import { BinaryOperator, ExpressionType } from '../types/ExpressionType';
import type { LinqStatement, TSqlStatement } from '../types/StatementTypes';

// LINQ expressions
import { LinqQueryExpression } from '../linqExpressions/LinqQueryExpression';
import { LinqFromExpression } from '../linqExpressions/LinqFromExpression';
import { LinqJoinExpression } from '../linqExpressions/LinqJoinExpression';
import { LinqWhereExpression } from '../linqExpressions/LinqWhereExpression';
import { LinqGroupByExpression } from '../linqExpressions/LinqGroupByExpression';
import { LinqHavingExpression } from '../linqExpressions/LinqHavingExpression';
import { LinqOrderByExpression } from '../linqExpressions/LinqOrderByExpression';
import { LinqSelectExpression } from '../linqExpressions/LinqSelectExpression';
import { LinqDropTableExpression } from '../linqExpressions/LinqDropTableExpression';
import { LinqDeleteExpression } from '../linqExpressions/LinqDeleteExpression';

// T-SQL expressions
import { QueryExpression } from '../expressions/QueryExpression';
import { FromExpression } from '../expressions/FromExpression';
import { JoinExpression } from '../expressions/JoinExpression';
import { WhereExpression } from '../expressions/WhereExpression';
import { GroupByExpression } from '../expressions/GroupByExpression';
import { HavingExpression } from '../expressions/HavingExpression';
import { OrderByExpression, OrderByItem } from '../expressions/OrderByExpression';
import { SelectExpression, SelectItem } from '../expressions/SelectExpression';
import { DropTableExpression } from '../expressions/DropTableExpression';
import { DeleteExpression } from '../expressions/DeleteExpression';
import { ColumnExpression } from '../expressions/ColumnExpression';
import { LiteralExpression } from '../expressions/LiteralExpression';
import { BinaryExpression } from '../expressions/BinaryExpression';
import { UnaryExpression } from '../expressions/UnaryExpression';
import { FunctionExpression } from '../expressions/FunctionExpression';

// Converter from LINQ expressions to T-SQL expressions
export class LinqToTSqlConverter {
  
  private formatTableName(
    tableName: string, 
    databaseName?: string, 
    schemaName?: string
  ): string {
    if (databaseName && schemaName) {
      return `${databaseName}.${schemaName}.${tableName}`;
    } else if (databaseName) {
      return `${databaseName}.${tableName}`;
    }
    return tableName;
  }
  
  // Convert LINQ query to T-SQL query
  convert(linqQuery: LinqStatement): TSqlStatement {
    switch (linqQuery.type) {
      case ExpressionType.LinqDropTable: {
        const dropTable = linqQuery as LinqDropTableExpression;
        const tableName = this.formatTableName(
          dropTable.tableName,
          dropTable.databaseName,
          dropTable.schemaName
        );
        return new DropTableExpression(tableName);
      }
      
      case ExpressionType.LinqDelete: {
        const deleteExpr = linqQuery as LinqDeleteExpression;
        const tableName = this.formatTableName(
          deleteExpr.tableName,
          deleteExpr.databaseName,
          deleteExpr.schemaName
        );
        
        const where = deleteExpr.whereCondition 
          ? new WhereExpression(deleteExpr.whereCondition)
          : undefined;
        
        return new DeleteExpression(
          tableName,
          where,
          deleteExpr.topCount,
          deleteExpr.isPercent
        );
      }
      
      case ExpressionType.LinqQuery: {
        // LinqQuery 的轉換邏輯
        const query = linqQuery as LinqQueryExpression;
        
        // Convert FROM
        const from = query.from 
          ? new FromExpression(
              this.formatTableName(
                query.from.tableName,
                query.from.databaseName,
                query.from.schemaName
              ),
              query.from.alias,
              query.from.hints
            )
          : undefined;
        
        // Convert JOINs
        const joins = query.joins.map(j => 
          new JoinExpression(
            j.joinType,
            this.formatTableName(
              j.tableName,
              j.databaseName,
              j.schemaName
            ),
            this.convertExpression(j.condition),
            j.alias,
            j.hints
          )
        );
        
        // Convert WHEREs - combine multiple WHERE clauses with AND
        let where: WhereExpression | undefined;
        if (query.wheres.length > 0) {
          let combinedCondition: Expression = this.convertExpression(query.wheres[0].condition);
          
          for (let i = 1; i < query.wheres.length; i++) {
            const nextCondition = this.convertExpression(query.wheres[i].condition);
            combinedCondition = new BinaryExpression(combinedCondition, BinaryOperator.And, nextCondition);
          }
          
          where = new WhereExpression(combinedCondition);
        }
        
        // Convert GROUP BYs - combine multiple GROUP BY clauses
        let groupBy: GroupByExpression | undefined;
        if (query.groupBys.length > 0) {
          const allColumns: Expression[] = [];
          for (const gb of query.groupBys) {
            allColumns.push(...gb.columns.map(c => this.convertExpression(c)));
          }
          groupBy = new GroupByExpression(allColumns);
        }
        
        // Convert HAVING
        const having = query.having 
          ? new HavingExpression(this.convertExpression(query.having.condition))
          : undefined;
        
        // Convert ORDER BYs - combine multiple ORDER BY clauses
        let orderBy: OrderByExpression | undefined;
        if (query.orderBys.length > 0) {
          const allItems: OrderByItem[] = [];
          for (const ob of query.orderBys) {
            allItems.push(...ob.items.map(item => ({
              expression: this.convertExpression(item.expression),
              direction: item.direction
            })));
          }
          orderBy = new OrderByExpression(allItems);
        }
        
        // Convert SELECT
        const select = query.select 
          ? new SelectExpression(
              query.select.items.map(item => ({
                expression: this.convertExpression(item.expression),
                alias: item.alias
              })),
              query.select.isDistinct,
              query.select.topCount
            )
          : undefined;
        
        return new QueryExpression(select, from, joins, where, groupBy, having, orderBy);
      }
      
      default: {
        const _exhaustive: never = linqQuery as never;
        throw new Error(`Unknown LINQ statement type`);
      }
    }
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

