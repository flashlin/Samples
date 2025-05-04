import { LinqIdentifierExpr, LinqParser } from './index';

describe('LinqParser', () => {
  it('should parse simple select', () => {
    const linq = new LinqParser();
    const expr = linq.parse('from tb1 in customer select tb1');
    expect(expr.From.Identifier).toBe('tb1');
    expect(expr.From.Source).toBe('customer');
    // 保證 expr.Select 不為 undefined
    expect(expr.Select).toBeDefined();
    expect((expr.Select!.Expression as LinqIdentifierExpr).Name).toBe('tb1');
  });
}); 