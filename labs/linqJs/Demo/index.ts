// @ts-ignore
// Import Hello function from linqjs
import { LinqQueryExpr, LinqFromExpr, LinqWhereExpr, LinqSelectExpr, LinqJoinExpr, LinqBinaryExpr, LinqIdentifierExpr, LinqMemberAccessExpr, LinqLiteralExpr } from 'linqjs';

// 範例客戶資料
const myCustomers = [
  { FirstName: "Chris", LastName: "Pearson", id: 1, status: "active" },
  { FirstName: "Kate", LastName: "Johnson", id: 2, status: "inactive" },
  { FirstName: "Josh", LastName: "Sutherland", id: 3, status: "active" },
  { FirstName: "John", LastName: "Ronald", id: 4, status: "inactive" },
  { FirstName: "Steve", LastName: "Pinkerton", id: 5, status: "active" }
];

// 測試訂單資料
const myOrders = [
  { OrderId: 101, CustomerId: 1, Amount: 250, Status: "shipped" },
  { OrderId: 102, CustomerId: 2, Amount: 120, Status: "pending" },
  { OrderId: 103, CustomerId: 1, Amount: 320, Status: "shipped" },
  { OrderId: 104, CustomerId: 3, Amount: 150, Status: "cancelled" },
  { OrderId: 105, CustomerId: 5, Amount: 500, Status: "shipped" }
];

// 查詢例子：查找狀態為 "active" 的客戶 join 訂單

// 建立查詢物件
const query = new LinqQueryExpr();

// 設定 FROM
query.From.Identifier = "c";
query.From.Source = "myCustomers";

// 設定 WHERE 條件：c.status == "active"
const whereExpr = new LinqWhereExpr();
const left = new LinqMemberAccessExpr();
left.Target = new LinqIdentifierExpr();
left.Target.Name = "c";
left.MemberName = "status";
const right = new LinqLiteralExpr();
right.Value = "active";
const cond = new LinqBinaryExpr();
cond.Left = left;
cond.Operator = "==";
cond.Right = right;
whereExpr.Condition = cond;
query.Where = whereExpr;

// 設定 JOIN myOrders o ON c.id == o.CustomerId
const join = new LinqJoinExpr();
join.Identifier = "o";
join.Source = "myOrders";
const outerKey = new LinqMemberAccessExpr();
outerKey.Target = new LinqIdentifierExpr();
outerKey.Target.Name = "c";
outerKey.MemberName = "id";
const innerKey = new LinqMemberAccessExpr();
innerKey.Target = new LinqIdentifierExpr();
innerKey.Target.Name = "o";
innerKey.MemberName = "CustomerId";
join.OuterKey = outerKey;
join.InnerKey = innerKey;
query.Joins.push(join);

// 設定 SELECT c, o
const select = new LinqSelectExpr();
select.Expression = new LinqIdentifierExpr();
select.Expression.Name = "{c, o}";
query.Select = select;

console.log('Linq 查詢物件：', JSON.stringify(query, null, 2)); 