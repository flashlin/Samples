// Forward declarations - will be imported from actual expression files
import type { QueryExpression } from '../expressions/QueryExpression';
import type { SelectExpression } from '../expressions/SelectExpression';
import type { FromExpression } from '../expressions/FromExpression';
import type { JoinExpression } from '../expressions/JoinExpression';
import type { WhereExpression } from '../expressions/WhereExpression';
import type { GroupByExpression } from '../expressions/GroupByExpression';
import type { HavingExpression } from '../expressions/HavingExpression';
import type { OrderByExpression } from '../expressions/OrderByExpression';
import type { DropTableExpression } from '../expressions/DropTableExpression';
import type { DeleteExpression } from '../expressions/DeleteExpression';
import type { ColumnExpression } from '../expressions/ColumnExpression';
import type { LiteralExpression } from '../expressions/LiteralExpression';
import type { BinaryExpression } from '../expressions/BinaryExpression';
import type { UnaryExpression } from '../expressions/UnaryExpression';
import type { FunctionExpression } from '../expressions/FunctionExpression';

// LINQ expressions
import type { LinqQueryExpression } from '../linqExpressions/LinqQueryExpression';
import type { LinqFromExpression } from '../linqExpressions/LinqFromExpression';
import type { LinqJoinExpression } from '../linqExpressions/LinqJoinExpression';
import type { LinqWhereExpression } from '../linqExpressions/LinqWhereExpression';
import type { LinqGroupByExpression } from '../linqExpressions/LinqGroupByExpression';
import type { LinqHavingExpression } from '../linqExpressions/LinqHavingExpression';
import type { LinqOrderByExpression } from '../linqExpressions/LinqOrderByExpression';
import type { LinqSelectExpression } from '../linqExpressions/LinqSelectExpression';
import type { LinqDropTableExpression } from '../linqExpressions/LinqDropTableExpression';
import type { LinqDeleteExpression } from '../linqExpressions/LinqDeleteExpression';

// Visitor pattern interface
export interface ExpressionVisitor<T> {
  // T-SQL Query Expressions
  visitQuery(expr: QueryExpression): T;
  visitSelect(expr: SelectExpression): T;
  visitFrom(expr: FromExpression): T;
  visitJoin(expr: JoinExpression): T;
  visitWhere(expr: WhereExpression): T;
  visitGroupBy(expr: GroupByExpression): T;
  visitHaving(expr: HavingExpression): T;
  visitOrderBy(expr: OrderByExpression): T;
  visitDropTable(expr: DropTableExpression): T;
  visitDelete(expr: DeleteExpression): T;
  
  // T-SQL Condition and Operation Expressions
  visitColumn(expr: ColumnExpression): T;
  visitLiteral(expr: LiteralExpression): T;
  visitBinary(expr: BinaryExpression): T;
  visitUnary(expr: UnaryExpression): T;
  visitFunction(expr: FunctionExpression): T;
  
  // LINQ Query Expressions
  visitLinqQuery(expr: LinqQueryExpression): T;
  visitLinqFrom(expr: LinqFromExpression): T;
  visitLinqJoin(expr: LinqJoinExpression): T;
  visitLinqWhere(expr: LinqWhereExpression): T;
  visitLinqGroupBy(expr: LinqGroupByExpression): T;
  visitLinqHaving(expr: LinqHavingExpression): T;
  visitLinqOrderBy(expr: LinqOrderByExpression): T;
  visitLinqSelect(expr: LinqSelectExpression): T;
  visitLinqDropTable(expr: LinqDropTableExpression): T;
  visitLinqDelete(expr: LinqDeleteExpression): T;
}

