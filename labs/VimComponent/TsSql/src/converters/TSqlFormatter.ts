import { ExpressionVisitor } from '../types/ExpressionVisitor';
import { Expression } from '../types/BaseExpression';

// T-SQL expressions
import { QueryExpression } from '../expressions/QueryExpression';
import { SelectExpression } from '../expressions/SelectExpression';
import { FromExpression } from '../expressions/FromExpression';
import { JoinExpression } from '../expressions/JoinExpression';
import { WhereExpression } from '../expressions/WhereExpression';
import { GroupByExpression } from '../expressions/GroupByExpression';
import { HavingExpression } from '../expressions/HavingExpression';
import { OrderByExpression } from '../expressions/OrderByExpression';
import { DropTableExpression } from '../expressions/DropTableExpression';
import { DeleteExpression } from '../expressions/DeleteExpression';
import { ColumnExpression } from '../expressions/ColumnExpression';
import { LiteralExpression } from '../expressions/LiteralExpression';
import { BinaryExpression } from '../expressions/BinaryExpression';
import { UnaryExpression } from '../expressions/UnaryExpression';
import { FunctionExpression } from '../expressions/FunctionExpression';

// LINQ expressions (not formatted, but need to handle in visitor)
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

// T-SQL Formatter - converts T-SQL expressions to formatted SQL string
export class TSqlFormatter implements ExpressionVisitor<string> {
  private indent: number = 0;
  private indentStr: string = '  '; // 2 spaces
  
  // Format a query expression
  format(expr: Expression): string {
    return expr.accept(this);
  }
  
  // Visit Query Expression
  visitQuery(expr: QueryExpression): string {
    const parts: string[] = [];
    
    // SELECT clause
    if (expr.select) {
      parts.push(expr.select.accept(this));
    }
    
    // FROM clause
    if (expr.from) {
      parts.push(expr.from.accept(this));
    }
    
    // JOIN clauses
    for (const join of expr.joins) {
      parts.push(join.accept(this));
    }
    
    // WHERE clause
    if (expr.where) {
      parts.push(expr.where.accept(this));
    }
    
    // GROUP BY clause
    if (expr.groupBy) {
      parts.push(expr.groupBy.accept(this));
    }
    
    // HAVING clause
    if (expr.having) {
      parts.push(expr.having.accept(this));
    }
    
    // ORDER BY clause
    if (expr.orderBy) {
      parts.push(expr.orderBy.accept(this));
    }
    
    return parts.filter(p => p.length > 0).join('\n');
  }
  
  // Visit SELECT Expression
  visitSelect(expr: SelectExpression): string {
    const top = expr.topCount ? `TOP ${expr.topCount} ` : '';
    const distinct = expr.isDistinct ? 'DISTINCT ' : '';
    const items = expr.items.map(item => {
      const exprStr = item.expression.accept(this);
      return item.alias ? `${exprStr} AS ${item.alias}` : exprStr;
    }).join(', ');
    
    return `SELECT ${top}${distinct}${items}`;
  }
  
  // Visit FROM Expression
  visitFrom(expr: FromExpression): string {
    const alias = expr.alias ? ` ${expr.alias}` : '';
    const hints = expr.hints && expr.hints.length > 0 
      ? ` WITH(${expr.hints.join(', ')})` 
      : '';
    return `FROM ${expr.tableName}${alias}${hints}`;
  }
  
  // Visit JOIN Expression
  visitJoin(expr: JoinExpression): string {
    const alias = expr.alias ? ` ${expr.alias}` : '';
    const hints = expr.hints && expr.hints.length > 0 
      ? ` WITH(${expr.hints.join(', ')})` 
      : '';
    const condition = expr.condition.accept(this);
    return `${expr.joinType} JOIN ${expr.tableName}${alias}${hints} ON ${condition}`;
  }
  
  // Visit WHERE Expression
  visitWhere(expr: WhereExpression): string {
    const condition = expr.condition.accept(this);
    
    // Check if condition is complex (contains AND/OR)
    if (this.isComplexCondition(expr.condition)) {
      return `WHERE ${this.formatComplexCondition(expr.condition)}`;
    }
    
    return `WHERE ${condition}`;
  }
  
  // Visit GROUP BY Expression
  visitGroupBy(expr: GroupByExpression): string {
    const columns = expr.columns.map(c => c.accept(this)).join(', ');
    return `GROUP BY ${columns}`;
  }
  
  // Visit HAVING Expression
  visitHaving(expr: HavingExpression): string {
    const condition = expr.condition.accept(this);
    return `HAVING ${condition}`;
  }
  
  // Visit ORDER BY Expression
  visitOrderBy(expr: OrderByExpression): string {
    const items = expr.items.map(item => {
      const exprStr = item.expression.accept(this);
      return `${exprStr} ${item.direction}`;
    }).join(', ');
    
    return `ORDER BY ${items}`;
  }
  
  // Visit Column Expression
  visitColumn(expr: ColumnExpression): string {
    return expr.fullName;
  }
  
  // Visit Literal Expression
  visitLiteral(expr: LiteralExpression): string {
    if (expr.literalType === 'string') {
      return `'${expr.value}'`;
    } else if (expr.literalType === 'null') {
      return 'NULL';
    } else if (expr.literalType === 'boolean') {
      return expr.value ? '1' : '0';
    }
    return String(expr.value);
  }
  
  // Visit Binary Expression
  visitBinary(expr: BinaryExpression): string {
    const left = expr.left.accept(this);
    const right = expr.right.accept(this);
    return `${left} ${expr.operator} ${right}`;
  }
  
  // Visit Unary Expression
  visitUnary(expr: UnaryExpression): string {
    const operand = expr.operand.accept(this);
    
    if (expr.operator === 'IS NULL' || expr.operator === 'IS NOT NULL') {
      return `${operand} ${expr.operator}`;
    }
    
    return `${expr.operator} ${operand}`;
  }
  
  // Visit Function Expression
  visitFunction(expr: FunctionExpression): string {
    const args = expr.args.map(a => a.accept(this)).join(', ');
    const funcStr = `${expr.functionName.toUpperCase()}(${args})`;
    return expr.alias ? `${funcStr} AS ${expr.alias}` : funcStr;
  }
  
  // Visit DROP TABLE Expression
  visitDropTable(expr: DropTableExpression): string {
    return `DROP TABLE ${expr.tableName}`;
  }
  
  visitDelete(expr: DeleteExpression): string {
    let sql = 'DELETE';
    
    if (expr.topCount !== undefined) {
      sql += ` TOP (${expr.topCount})`;
      if (expr.isPercent) {
        sql += ' PERCENT';
      }
    }
    
    sql += ` FROM ${expr.tableName}`;
    
    if (expr.where) {
      sql += ` WHERE ${expr.where.condition.accept(this)}`;
    }
    
    return sql;
  }
  
  // LINQ expressions (should not be formatted, but need to handle)
  visitLinqQuery(expr: LinqQueryExpression): string {
    throw new Error('Cannot format LINQ query directly. Convert to T-SQL first.');
  }
  
  visitLinqFrom(expr: LinqFromExpression): string {
    throw new Error('Cannot format LINQ expression directly. Convert to T-SQL first.');
  }
  
  visitLinqJoin(expr: LinqJoinExpression): string {
    throw new Error('Cannot format LINQ expression directly. Convert to T-SQL first.');
  }
  
  visitLinqWhere(expr: LinqWhereExpression): string {
    throw new Error('Cannot format LINQ expression directly. Convert to T-SQL first.');
  }
  
  visitLinqGroupBy(expr: LinqGroupByExpression): string {
    throw new Error('Cannot format LINQ expression directly. Convert to T-SQL first.');
  }
  
  visitLinqHaving(expr: LinqHavingExpression): string {
    throw new Error('Cannot format LINQ expression directly. Convert to T-SQL first.');
  }
  
  visitLinqOrderBy(expr: LinqOrderByExpression): string {
    throw new Error('Cannot format LINQ expression directly. Convert to T-SQL first.');
  }
  
  visitLinqSelect(expr: LinqSelectExpression): string {
    throw new Error('Cannot format LINQ expression directly. Convert to T-SQL first.');
  }
  
  visitLinqDropTable(expr: LinqDropTableExpression): string {
    throw new Error('Cannot format LINQ DROP TABLE directly. Convert to T-SQL first.');
  }
  
  visitLinqDelete(expr: LinqDeleteExpression): string {
    throw new Error('Cannot format LINQ DELETE directly. Convert to T-SQL first.');
  }
  
  // Helper methods
  private isComplexCondition(expr: Expression): boolean {
    if (expr instanceof BinaryExpression) {
      return expr.operator === 'AND' || expr.operator === 'OR';
    }
    return false;
  }
  
  private formatComplexCondition(expr: Expression): string {
    if (expr instanceof BinaryExpression && (expr.operator === 'AND' || expr.operator === 'OR')) {
      const left = this.isComplexCondition(expr.left) 
        ? this.formatComplexCondition(expr.left)
        : expr.left.accept(this);
      
      const right = this.isComplexCondition(expr.right)
        ? this.formatComplexCondition(expr.right)
        : expr.right.accept(this);
      
      // Add indentation for multi-line conditions
      if (expr.operator === 'AND') {
        return `${left}\n  ${expr.operator} ${right}`;
      } else {
        return `(${left}\n   ${expr.operator} ${right})`;
      }
    }
    
    return expr.accept(this);
  }
}

