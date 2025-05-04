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

  it('should parse join and select new', () => {
    const linq = new LinqParser();
    const expr = linq.parse('from tb1 in customer join tb2 in orders on tb2.CustomerId equals tb1.id select new { tb1.id, Name = tb1.LastName, tb2.Amount }');
    // 檢查 From
    expect(expr.From.Identifier).toBe('tb1');
    expect(expr.From.Source).toBe('customer');
    // 檢查 Join
    expect(expr.Joins.length).toBe(1);
    expect(expr.Joins[0].Identifier).toBe('tb2');
    expect(expr.Joins[0].Source).toBe('orders');
    // 驗證 join on 條件
    expect((expr.Joins[0] as any).OuterKeyRaw).toBe('tb2.CustomerId');
    expect((expr.Joins[0] as any).InnerKeyRaw).toBe('tb1.id');
    // 檢查 Select
    expect(expr.Select).toBeDefined();
    // 這裡僅檢查 select new 結構字串
    expect((expr.Select!.Expression as any).Raw).toBe('new { tb1.id, Name = tb1.LastName, tb2.Amount }');
  });
}); 