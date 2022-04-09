namespace SqliteCli.Repos
{
	public class ReportTranItem
	{
		[DisplayString("", 7)]
		public string TranType { get; set; }

		[DisplayString("", 9)]
		public string StockId { set; get; }

		[DisplayString("", 30)]
		public string StockName { set; get; }

		//2022-04-04
		//假如這邊 property 用 decimal or double
		//卻噴出 ERROR: Sqlite AVG(xxx) 出來是 double, but dapper 卻 parse to int64
		//只好改為 string
		[DisplayString("", 6)]
		public decimal MinStockPrice { get; set; }

		[DisplayString("", 6)]
		public decimal AvgStockPrice { get; set; }

		[DisplayString("", 6)]
		public decimal MaxStockPrice { get; set; }

		[DisplayString("", 7)]
		public int NumberOfShare { get; set; }

		[DecimalString(7)]
		public decimal HandlingFee { get; set; }

		[DecimalString(20)]
		public decimal Balance { get; set; }
		
		[DecimalString(20)]
		public decimal CurrentPrice { get; set; }
	}
}
