import { LinqQueryExpr, LinqFromExpr, LinqSelectExpr, LinqIdentifierExpr } from './index';

class LinqParser {
  // 解析 LINQ 查詢字串，回傳 AST
  public parse(query: string): LinqQueryExpr {
    // 這裡僅實作簡單的 from ... in ... select ... 範例
    // 實際可依需求擴充
    const match = query.match(/from\s+(\w+)\s+in\s+(\w+)\s+select\s+(\w+)/);
    if (!match) throw new Error('查詢語法錯誤');
    const fromId = match[1];
    const fromSrc = match[2];
    const selectId = match[3];
    const expr = new LinqQueryExpr();
    expr.From = new LinqFromExpr();
    expr.From.Identifier = fromId;
    expr.From.Source = fromSrc;
    expr.Select = new LinqSelectExpr();
    const idExpr = new LinqIdentifierExpr();
    idExpr.Name = selectId;
    expr.Select.Expression = idExpr;
    return expr;
  }
}

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