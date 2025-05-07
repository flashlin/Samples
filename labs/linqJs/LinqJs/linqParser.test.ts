import { LinqIdentifierExpr, LinqMemberAccessExpr } from './LinqExprs';
import { LinqParser } from './LinqParser';
import { LinqQueryExpr } from './LinqExprs';
import { LinqNewExpr } from './LinqExprs';

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
    const newExpr = expr.Select!.Expression as LinqNewExpr;
    expect(newExpr.Properties.length).toBe(3);
    // tb1.id
    expect(newExpr.Properties[0].Name).toBe('tb1');
    expect((newExpr.Properties[0].Value as LinqMemberAccessExpr).MemberName).toBe('id');
    // Name = tb1.LastName
    expect(newExpr.Properties[1].Name).toBe('Name');
    expect((newExpr.Properties[1].Value as LinqMemberAccessExpr).MemberName).toBe('LastName');
    // 驗證 tb1.LastName 的 Target.Name
    expect(((newExpr.Properties[1].Value as LinqMemberAccessExpr).Target as LinqIdentifierExpr).Name).toBe('tb1');
    // tb2.Amount
    expect(newExpr.Properties[2].Name).toBe('tb2');
    expect((newExpr.Properties[2].Value as LinqMemberAccessExpr).MemberName).toBe('Amount');
  });


  it('should parse join with on equals and where', () => {
    const linq = new LinqParser();
    const expr = linq.parse('from tb1 in customer join tb2 in orders on tb2.CustomerId equals tb1.id where tb1.status == 1 select tb1') as LinqQueryExpr;
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
    // 檢查 Where
    expect(expr.Where).toBeDefined();
    const cond = expr.Where!.Condition as any;
    // 只驗證型別與欄位
    expect(cond.Left.Target.Name).toBe('tb1');
    expect(cond.Left.MemberName).toBe('status');
    expect(cond.Operator).toBe('==');
    expect(cond.Right.Name).toBe('1');
    // 檢查 Select
    expect(expr.Select).toBeDefined();
    expect((expr.Select!.Expression as LinqIdentifierExpr).Name).toBe('tb1');
  });

  it('should parse group by and select new', () => {
    const linq = new LinqParser();
    const expr = linq.parse('from o in orders join c in customers on o.CustomerId equals c.CustomerId group o by c.CustomerId into g select new { CustomerId = g.Key, TotalAmount = g.Sum(o => o.Amount) }') as LinqQueryExpr;
    // 檢查 From
    expect(expr.From.Identifier).toBe('o');
    expect(expr.From.Source).toBe('orders');
    // 檢查 Join
    expect(expr.Joins.length).toBe(1);
    expect(expr.Joins[0].Identifier).toBe('c');
    expect(expr.Joins[0].Source).toBe('customers');
    // 驗證 join on 條件
    const outerKey = expr.Joins[0].OuterKey as LinqMemberAccessExpr;
    const innerKey = expr.Joins[0].InnerKey as LinqMemberAccessExpr;
    expect((outerKey.Target as LinqIdentifierExpr).Name).toBe('o');
    expect(outerKey.MemberName).toBe('CustomerId');
    expect((innerKey.Target as LinqIdentifierExpr).Name).toBe('c');
    expect(innerKey.MemberName).toBe('CustomerId');
    // 驗證 group by
    expect(expr.Group).toBeDefined();
    const groupKey = expr.Group!.Key as LinqMemberAccessExpr;
    expect((groupKey.Target as LinqIdentifierExpr).Name).toBe('c');
    expect(groupKey.MemberName).toBe('CustomerId');
    // 驗證 select new
    expect(expr.Select).toBeDefined();
    // 驗證 select new 結構 AST
    const newExpr = expr.Select!.Expression as LinqNewExpr;
    expect(newExpr.Properties.length).toBe(2);
    // CustomerId = g.Key
    expect(newExpr.Properties[0].Name).toBe('CustomerId');
    const customerIdValue = newExpr.Properties[0].Value as LinqMemberAccessExpr;
    expect(customerIdValue.MemberName).toBe('Key');
    // TotalAmount = g.Sum(o => o.Amount)
    expect(newExpr.Properties[1].Name).toBe('TotalAmount');
    const sumExpr = newExpr.Properties[1].Value as any;
    expect(sumExpr.FunctionName).toBe('Sum');
    expect(sumExpr.Arguments.length).toBe(1);
    const arg0 = sumExpr.Arguments[0] as LinqMemberAccessExpr;
    expect((arg0.Target as LinqIdentifierExpr).Name).toBe('o');
    expect(arg0.MemberName).toBe('Amount');
  });
}); 