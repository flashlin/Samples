import { parseLinq } from '../linq'
import { SelectClause, QueryBody, FromClause, WhereClause, IdentifierExpression } from '../linqExpressions'

describe('parseLinq', () => {
  it('should parse simple from-where-select', () => {
    const ast = parseLinq('from x in xs where y select z')
    // 檢查 select
    expect(ast.kind).toBe('SelectClause')
    const select = ast as SelectClause
    expect(select.expression.kind).toBe('QueryBody')
    const queryBody = select.expression as QueryBody
    expect(queryBody.clauses.length).toBe(2)
    // from
    const from = queryBody.clauses[0] as FromClause
    expect(from.kind).toBe('FromClause')
    expect(from.identifier).toBe('x')
    expect((from.source as IdentifierExpression).name).toBe('xs')
    // where
    const where = queryBody.clauses[1] as WhereClause
    expect(where.kind).toBe('WhereClause')
    expect((where.condition as IdentifierExpression).name).toBe('y')
    // 驗證 select 子句內容
    const selectExpr = select.expression as QueryBody
    expect(selectExpr.kind).toBe('QueryBody')
  })

  it('should parse select only', () => {
    const ast = parseLinq('select foo')
    expect(ast.kind).toBe('SelectClause')
    expect((ast.expression as IdentifierExpression).name).toBe('foo')
  })
}) 