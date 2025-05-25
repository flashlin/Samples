import { textTokenize } from './textTokenizer'
import {
  LinqExpression,
  FromClause,
  WhereClause,
  SelectClause,
  IdentifierExpression,
  QueryBody,
} from './linqExpressions'

// 解析 Linq 查詢語法，回傳 SelectClause AST
export function parseLinq(text: string): SelectClause {
  const tokens = Array.from(textTokenize(text)).filter(t => t.trim() !== '')
  let index = 0

  function peek(): string | undefined {
    return tokens[index]
  }
  function next(): string | undefined {
    return tokens[index++]
  }
  function expect(val: string) {
    const t = next()
    if (t !== val) throw new Error(`Expected '${val}', got '${t}'`)
  }

  // parse identifier
  function parseIdentifier(): IdentifierExpression {
    const name = next()
    if (!name) throw new Error('Unexpected end, expected identifier')
    return { kind: 'IdentifierExpression', name }
  }

  // parse from ...
  function parseFrom(): FromClause {
    expect('from')
    const identifier = next()
    expect('in')
    const source = parseIdentifier()
    return {
      kind: 'FromClause',
      identifier: identifier!,
      source,
    }
  }

  // parse where ...
  function parseWhere(): WhereClause {
    expect('where')
    // 這裡簡化只支援單一 identifier
    const condition = parseIdentifier()
    return {
      kind: 'WhereClause',
      condition,
    }
  }

  // parse select ...
  function parseSelect(): SelectClause {
    expect('select')
    const expr = parseIdentifier()
    return {
      kind: 'SelectClause',
      expression: expr,
    }
  }

  // 主體解析
  const clauses: LinqExpression[] = []
  if (peek() === 'from') {
    clauses.push(parseFrom())
  }
  if (peek() === 'where') {
    clauses.push(parseWhere())
  }
  let selectClause: SelectClause | null = null
  if (peek() === 'select') {
    selectClause = parseSelect()
  }
  if (!selectClause) throw new Error('Linq must end with select')

  // 若有多個子句，包成 QueryBody
  if (clauses.length > 0) {
    const queryBody: QueryBody = {
      kind: 'QueryBody',
      clauses,
    }
    selectClause.expression = queryBody
  }
  return selectClause
} 