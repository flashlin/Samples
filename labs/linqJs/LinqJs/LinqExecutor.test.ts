import { LinqExecutor } from './LinqExecutor';
import { LinqParser } from './LinqParser';
import { LinqQueryExpr } from './LinqExprs';

describe('LinqExecutor', () => {
  let myCustomers: any[];
  let myOrders: any[];
  let linq: LinqExecutor;

  beforeEach(() => {
    // 範例客戶資料
    myCustomers = [
      { FirstName: 'Chris', LastName: 'Pearson', id: 1, status: 'active' },
      { FirstName: 'Kate', LastName: 'Johnson', id: 2, status: 'inactive' },
      { FirstName: 'Josh', LastName: 'Sutherland', id: 3, status: 'active' },
      { FirstName: 'John', LastName: 'Ronald', id: 4, status: 'inactive' },
      { FirstName: 'Steve', LastName: 'Pinkerton', id: 5, status: 'active' }
    ];
    // 測試訂單資料
    myOrders = [
      { OrderId: 101, CustomerId: 1, Amount: 250, Status: 'shipped' },
      { OrderId: 102, CustomerId: 2, Amount: 120, Status: 'pending' },
      { OrderId: 103, CustomerId: 1, Amount: 320, Status: 'shipped' },
      { OrderId: 104, CustomerId: 3, Amount: 150, Status: 'cancelled' },
      { OrderId: 105, CustomerId: 5, Amount: 500, Status: 'shipped' }
    ];
    // 建立 LinqExecutor 並設定資料來源
    linq = new LinqExecutor();
    linq.Data['customers'] = myCustomers;
    linq.Data['orders'] = myOrders;
  });

  it('orders left join customers 然後 select OrderId, Amount, LastName', () => {
    // 測試查詢
    const parser = new LinqParser();
    const query = 'from o in orders join c in customers on o.CustomerId equals c.id select new { o.OrderId, o.Amount, c.LastName }';
    const queryExpr = parser.parse(query) as LinqQueryExpr;
    const result = linq.query(queryExpr);
    // 驗證結果
    expect(result).toEqual([
      { OrderId: 101, Amount: 250, LastName: 'Pearson' },
      { OrderId: 102, Amount: 120, LastName: 'Johnson' },
      { OrderId: 103, Amount: 320, LastName: 'Pearson' },
      { OrderId: 104, Amount: 150, LastName: 'Sutherland' },
      { OrderId: 105, Amount: 500, LastName: 'Pinkerton' }
    ]);
  });
}); 