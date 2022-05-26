namespace SqliteCli.Repos
{

	public class ReportProfitItem
	{
		[DisplayString("", 9)] public string StockId { set; get; }

		[DisplayString("", 26)] public string StockName { set; get; }

		[DisplayString("yyyy/MM/dd", 10)] public DateTime Date { get; set; }
		
		[AmountString(6)]
		public decimal StockPrice { get; set; }
		
		[AmountString(11)]
		public decimal Profit { get; set; }
		
	}


	public class ReportTranItem : IConsoleTextProvider
	{
		[DisplayString("", 7)]
		public string TranType { get; set; }

		[DisplayString("", 9)]
		public string StockId { set; get; }

		[DisplayString("", 26)]
		public string StockName { set; get; }

		[DisplayString("yyyy/MM/dd", 10)]
		public DateTime MinTranTime { get; set; }

		//2022-04-04
		//假如這邊 property 用 decimal or double
		//卻噴出 ERROR: Sqlite AVG(xxx) 出來是 double, but dapper 卻 parse to int64
		//只好改為 string
		[DecimalString(6)]
		public decimal MinStockPrice { get; set; }

		[DecimalString(6)]
		public decimal AvgStockPrice { get; set; }

		[DecimalString(6)]
		public decimal MaxStockPrice { get; set; }

		[AmountString(7)]
		public int NumberOfShare { get; set; }

		[DecimalString(7)]
		public decimal HandlingFee { get; set; }

		[AmountString(12)]
		public decimal Balance { get; set; }

		[DecimalString(6)]
		public decimal CurrentPrice { get; set; }

		[AmountString(12)]
		public decimal CurrTotalPrice { get; set; }

		[AmountString(11)]
		public decimal Profit { get; set; }

		public ConsoleText GetConsoleText(string name, string value)
		{
			var foregroundColor = Console.ForegroundColor;
			var backgroundColor = Console.BackgroundColor;
			switch (name)
			{
				case nameof(StockId):
				case nameof(StockName):
				case nameof(CurrentPrice):
					if (AvgStockPrice < CurrentPrice)
					{
						foregroundColor = ConsoleColor.Red;
					}
					if (AvgStockPrice > CurrentPrice)
					{
						foregroundColor = ConsoleColor.Green;
					}
					break;
				case nameof(Profit):
					if (Profit < 0)
					{
						foregroundColor = ConsoleColor.Green;
					}
					if (Profit > 0)
					{
						foregroundColor = ConsoleColor.Red;
					}
					break;
			}

			return new ConsoleText
			{
				ForegroundColor = foregroundColor,
				BackgroundColor = backgroundColor,
				Text = value
			};
		}
	}
}
