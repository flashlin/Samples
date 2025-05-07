import { LinqIdentifierExpr, LinqMemberAccessExpr } from './LinqExprs';
import { LinqParser } from './LinqParser';
import { LinqQueryExpr } from './LinqExprs';

describe('LinqParser', () => {
  it('should parse simple select', () => {
    const linq = new LinqParser();
    const expr = linq.parse('from tb1 in customer select tb1') as LinqQueryExpr;
    expect(expr.From.Identifier).toBe('tb1');
    expect(expr.From.Source).toBe('customer');
    // 保證 expr.Select 不為 undefined
    expect(expr.Select).toBeDefined();
    expect((expr.Select!.Expression as LinqIdentifierExpr).Name).toBe('tb1');
  });

  it('should parse join and select new', () => {
    const linq = new LinqParser();
    const expr = linq.parse('from tb1 in customer join tb2 in orders on tb2.CustomerId equals tb1.id select new { tb1.id, Name = tb1.LastName, tb2.Amount }') as LinqQueryExpr;
    // 檢查 From
    expect(expr.From.Identifier).toBe('tb1');
    expect(expr.From.Source).toBe('customer');
    // 檢查 Join
    expect(expr.Joins.length).toBe(1);
    expect(expr.Joins[0].Identifier).toBe('tb2');
    expect(expr.Joins[0].Source).toBe('orders');
    // 驗證 join on 條件
    const outerKey = expr.Joins[0].OuterKey as LinqMemberAccessExpr;
    const innerKey = expr.Joins[0].InnerKey as LinqMemberAccessExpr;
    expect((outerKey.Target as LinqIdentifierExpr).Name).toBe('tb2');
    expect(outerKey.MemberName).toBe('CustomerId');
    expect((innerKey.Target as LinqIdentifierExpr).Name).toBe('tb1');
    expect(innerKey.MemberName).toBe('id');
    // 檢查 Select
    expect(expr.Select).toBeDefined();
    // 驗證 select new 結構 AST
    const newExpr = expr.Select!.Expression as any;
    expect(newExpr.Properties.length).toBe(3);
    // tb1.id
    expect(newExpr.Properties[0].Name).toBe('tb1');
    expect(newExpr.Properties[0].Value.MemberName).toBe('id');
    // Name = tb1.LastName
    expect(newExpr.Properties[1].Name).toBe('Name');
    expect(newExpr.Properties[1].Value.MemberName).toBe('LastName');
    // 驗證 tb1.LastName 的 Target.Name
    expect((newExpr.Properties[1].Value.Target as LinqIdentifierExpr).Name).toBe('tb1');
    // tb2.Amount
    expect(newExpr.Properties[2].Name).toBe('tb2');
    expect(newExpr.Properties[2].Value.MemberName).toBe('Amount');
  });
}); 