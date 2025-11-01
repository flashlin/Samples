// Expression type enumeration for all T-SQL and LINQ expressions
export enum ExpressionType {
  // T-SQL Query Expressions
  Query = 'Query',
  Select = 'Select',
  From = 'From',
  Join = 'Join',
  Where = 'Where',
  GroupBy = 'GroupBy',
  Having = 'Having',
  OrderBy = 'OrderBy',
  DropTable = 'DropTable',
  
  // T-SQL Condition and Operation Expressions
  Column = 'Column',
  Literal = 'Literal',
  Binary = 'Binary',
  Unary = 'Unary',
  Function = 'Function',
  
  // LINQ Query Expressions
  LinqQuery = 'LinqQuery',
  LinqFrom = 'LinqFrom',
  LinqJoin = 'LinqJoin',
  LinqWhere = 'LinqWhere',
  LinqGroupBy = 'LinqGroupBy',
  LinqHaving = 'LinqHaving',
  LinqOrderBy = 'LinqOrderBy',
  LinqSelect = 'LinqSelect',
  LinqDropTable = 'LinqDropTable',
}

// Join types
export enum JoinType {
  Inner = 'INNER',
  Left = 'LEFT',
  Right = 'RIGHT',
  Full = 'FULL',
  Cross = 'CROSS',
}

// Binary operators
export enum BinaryOperator {
  // Comparison
  Equal = '=',
  NotEqual = '<>',
  GreaterThan = '>',
  LessThan = '<',
  GreaterThanOrEqual = '>=',
  LessThanOrEqual = '<=',
  
  // Logical
  And = 'AND',
  Or = 'OR',
  
  // Arithmetic
  Add = '+',
  Subtract = '-',
  Multiply = '*',
  Divide = '/',
  Modulo = '%',
  
  // String
  Concat = '+',
  
  // Pattern matching
  Like = 'LIKE',
  NotLike = 'NOT LIKE',
  In = 'IN',
  NotIn = 'NOT IN',
}

// Unary operators
export enum UnaryOperator {
  Not = 'NOT',
  IsNull = 'IS NULL',
  IsNotNull = 'IS NOT NULL',
  Exists = 'EXISTS',
  NotExists = 'NOT EXISTS',
}

// Order direction
export enum OrderDirection {
  Asc = 'ASC',
  Desc = 'DESC',
}

