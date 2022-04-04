
using SqliteCli.Entities;

var db = new StockRepo();

var rc = db.QueryTrans(@"select * from trans");
foreach (var item in rc)
{
	Console.WriteLine(item.ToString());
}

Console.WriteLine("=== END ===");
