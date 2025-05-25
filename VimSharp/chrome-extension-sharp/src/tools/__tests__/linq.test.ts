import { parseLinq } from '../linq'

describe('parseLinq', () => {
  it('should parse simple from-where-select', () => {
    const ast = parseLinq('from x in xs where y select z')
    expect(ast).toEqual({
      kind: 'SelectClause',
      expression: {
        kind: 'QueryBody',
        clauses: [
          {
            kind: 'FromClause',
            identifier: 'x',
            source: { kind: 'IdentifierExpression', name: 'xs' }
          },
          {
            kind: 'WhereClause',
            condition: { kind: 'IdentifierExpression', name: 'y' }
          }
        ]
      },
      fields: [{ kind: 'IdentifierExpression', name: 'z' }]
    })
  })

  it('should parse select only', () => {
    const ast = parseLinq('select foo')
    expect(ast).toEqual({
      kind: 'SelectClause',
      expression: { kind: 'IdentifierExpression', name: 'foo' },
      fields: [{ kind: 'IdentifierExpression', name: 'foo' }]
    })
  })
}) 