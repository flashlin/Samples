import type { LinqQueryExpression } from '../linqExpressions/LinqQueryExpression';
import type { LinqDropTableExpression } from '../linqExpressions/LinqDropTableExpression';
import type { LinqDeleteExpression } from '../linqExpressions/LinqDeleteExpression';
import type { QueryExpression } from '../expressions/QueryExpression';
import type { DropTableExpression } from '../expressions/DropTableExpression';
import type { DeleteExpression } from '../expressions/DeleteExpression';

export type LinqStatement = 
  | LinqQueryExpression 
  | LinqDropTableExpression 
  | LinqDeleteExpression;

export type TSqlStatement = 
  | QueryExpression 
  | DropTableExpression 
  | DeleteExpression;

