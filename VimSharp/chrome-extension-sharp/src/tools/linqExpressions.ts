// C# Linq Query Syntax AST interfaces

// 基礎介面
export interface LinqExpression {
  kind: string
}

// from 語句
export interface FromClause extends LinqExpression {
  kind: 'FromClause'
  identifier: string
  source: LinqExpression
}

// let 語句
export interface LetClause extends LinqExpression {
  kind: 'LetClause'
  identifier: string
  expression: LinqExpression
}

// where 語句
export interface WhereClause extends LinqExpression {
  kind: 'WhereClause'
  condition: LinqExpression
}

// join 語句
export interface JoinClause extends LinqExpression {
  kind: 'JoinClause'
  identifier: string
  inExpression: LinqExpression
  onLeft: LinqExpression
  onRight: LinqExpression
  equals: boolean
  into?: string
}

// orderby 語句
export interface OrderByClause extends LinqExpression {
  kind: 'OrderByClause'
  expressions: OrderByExpression[]
}

export interface OrderByExpression extends LinqExpression {
  kind: 'OrderByExpression'
  expression: LinqExpression
  direction: 'asc' | 'desc'
}

// select 語句
export interface SelectClause extends LinqExpression {
  kind: 'SelectClause'
  expression: LinqExpression
  fields?: LinqExpression[]
}

// group 語句
export interface GroupClause extends LinqExpression {
  kind: 'GroupClause'
  groupExpression: LinqExpression
  byExpression: LinqExpression
}

// query body (多個子句)
export interface QueryBody extends LinqExpression {
  kind: 'QueryBody'
  clauses: LinqExpression[]
}

// identifier/常數/運算式等
export interface IdentifierExpression extends LinqExpression {
  kind: 'IdentifierExpression'
  name: string
}

export interface ConstantExpression extends LinqExpression {
  kind: 'ConstantExpression'
  value: any
}

export interface BinaryExpression extends LinqExpression {
  kind: 'BinaryExpression'
  left: LinqExpression
  operator: string
  right: LinqExpression
} 